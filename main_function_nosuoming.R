# ======== 环境配置 ========

# 加载必要的包
suppressPackageStartupMessages({
  library(tidyverse)    
  library(glmnet)       
  library(keras)        
  library(tensorflow)   
  library(xgboost)      
  library(caret)        
  library(gridExtra)    
  library(progress)     
  library(lubridate)   
  library(dplyr)
  library(iml)
  library(ggplot2)
  library(igraph)
  library(RColorBrewer)
  library(arrow)
  library(ggraph)
  library(patchwork)
  library(scales)
  library(ggrepel)
  library(data.table)
})

# 全局配置
config <- list(
  # 路径配置
  paths = list(
    base_dir = ".",
    data_dir = "data",
    output_dir = "output11",
    model_dir = "models",          
    plot_dir = "plots",            
    feature_cache = "features"     
  ),
  
  # 模型参数
  model_params = list(
    # Lasso模型参数
    lasso = list(
      alpha = 1,
      nfolds = 5,
      standardize = TRUE,
      lambda_rule = "lambda.min"
    ),
    
    # MLP模型参数
    mlp = list(
      layers = c(64, 32, 16),
      activation = "selu",
      dropout = c(0.3, 0.2, 0.1),
      batch_norm = TRUE,
      learning_rate = 0.001,
      batch_size = 16,
      epochs = 300,
      validation_split = 0.2,
      early_stopping = list(
        monitor = "val_loss",
        patience = 30,
        restore_best_weights = TRUE,
        min_delta = 1e-4
      ),
      l2_reg = 0.001
    ),
    
    # XGBoost模型参数
    xgboost = list(
      eta = 0.1,
      max_depth = 6,
      subsample = 0.8,
      colsample_bytree = 0.8,
      nrounds = 1000,
      early_stopping_rounds = 20,
      objective = "reg:squarederror",
      eval_metric = "rmse",
      nthread = 2,
      verbose = 0
    )
  ),
  
  # 模型选择策略
  model_selection = list(
    lasso_lambda_rule = "lambda.min",
    mlp_save_best = TRUE,
    xgb_early_stop = TRUE
  ),
  
  # 评估标准
  evaluation = list(
    r2_method = "traditional",
    test_eval_metrics = c("rmse", "r2", "mae", "mape"),
    importance_plot_topn = 15,
    save_predictions = TRUE
  ),
  
  # 交叉验证配置
  cv_params = list(
    k_folds = 5,
    random_seed = 42,
    stratified = FALSE
  ),
  
  # 图形输出配置
  plotting = list(
    theme = "minimal",
    color_palette = "viridis",
    dpi = 300,
    width = 10,
    height = 8,
    save_format = "png",
    importance_color = "dodgerblue4",
    line_color = "steelblue",
    point_color = "firebrick3",
    show_grid = FALSE,
    font_family = "sans",
    watermark = FALSE
  ),
  
  # 特征工程设置
  feature_engineering = list(
    max_interaction_order = 2,            # 最大交互特征阶数
    cache_features = TRUE,                # 是否缓存生成的特征
    feature_selection = TRUE,             # 是否进行特征选择
    top_features = 100                    # 特征选择保留数量
  )
)

# ======== 文件系统和路径管理 ========

#' 创建所有必要的目录
#' @param config 配置列表
#' @return 包含完整路径的列表
setup_directories <- function(config) {
  # 构建完整路径
  full_paths <- list(
    output_dir = file.path(config$paths$base_dir, config$paths$output_dir),
    model_dir = file.path(config$paths$base_dir, config$paths$output_dir, data_file_name, config$paths$model_dir),
    plot_dir = file.path(config$paths$base_dir, config$paths$output_dir, data_file_name, config$paths$plot_dir),
    data_dir = file.path(config$paths$base_dir, config$paths$output_dir, data_file_name, config$paths$data_dir),
    feature_cache = file.path(config$paths$base_dir, config$paths$output_dir, data_file_name, config$paths$feature_cache)
  )
  
  # 创建目录
  for (dir_path in full_paths) {
    if (!dir.exists(dir_path)) {
      dir.create(dir_path, recursive = TRUE)
      message(sprintf("Created directory: %s", dir_path))
    }
  }
  
  return(full_paths)
}

#' 构建文件路径辅助函数
#' @param base_dir 基础目录
#' @param filename 文件名
#' @param extension 扩展名
#' @return 完整文件路径
build_path <- function(base_dir, filename, extension = NULL) {
  if (!is.null(extension)) {
    filename <- paste0(filename, ".", extension)
  }
  file.path(base_dir, filename)
}

# ======== 数据预处理流程 ========

#' 处理数据，将二元矩阵转换为特征表示
#' @param data 输入数据框
#' @param binary_cols 二元特征列索引
#' @param target_cols 目标变量列索引
#' @param replace_zeros_with 替换0的值
#' @return 处理后的数据列表
process_data <- function(data, binary_cols, target_cols, replace_zeros_with = -1) {
  binary_matrix <- as.matrix(data[, binary_cols, drop = FALSE])
  target_values <- as.matrix(data[, target_cols, drop = FALSE])
  
  binary_matrix[binary_matrix == 0] <- replace_zeros_with
  
  return(list(
    binary_matrix = binary_matrix, 
    target_values = target_values,
    n_samples = nrow(binary_matrix),
    n_features = ncol(binary_matrix),
    feature_names = colnames(binary_matrix),
    target_name = colnames(target_values),
    data_summary = summary(target_values)
  ))
}

#' 交互特征生成函数
#' @param data 输入数据框，包含二进制物种矩阵
#' @param max_order 最大交互阶数
#' @param cache_path 缓存路径，NULL表示不缓存
#' @param cache_id 缓存标识符
#' @return 包含交互特征的数据框
generate_interaction_features <- function(data, max_order = 2, cache_path = NULL, cache_id = NULL) {
  if (!is.null(cache_path) && !is.null(cache_id)) {
    cache_file <- file.path(cache_path, paste0(cache_id, "_order", max_order, ".rds"))
    if (file.exists(cache_file)) {
      message("Loading cached interaction features...")
      return(readRDS(cache_file))
    }
  }
  
  message(sprintf("Generating interaction features up to order %d...", max_order))
  start_time <- Sys.time()
  
  n_rows <- nrow(data)
  feature_names <- colnames(data)
  
  interaction_features <- list()
  feature_count <- 0
  
  for (order in 1:max_order) {
    combos <- combn(feature_names, order, simplify = FALSE)
    
    for (combo in combos) {
      submat <- data[, combo, drop = FALSE]
      
      feature <- ifelse(rowSums(submat == 1) == length(combo), 1, -1)

      feature_name <- paste(combo, collapse = "_x_")
      
      interaction_features[[feature_name]] <- feature
      feature_count <- feature_count + 1
      
      if (feature_count %% 5000 == 0) {
        message(sprintf("  Generated %d features...", feature_count))
      }
    }
  }

  interaction_features_df <- as.data.frame(interaction_features)
  
  end_time <- Sys.time()
  elapsed <- difftime(end_time, start_time, units = "mins")
  message(sprintf("Generated %d interaction features in %.2f minutes", 
                  ncol(interaction_features_df), as.numeric(elapsed)))

  if (!is.null(cache_path) && !is.null(cache_id)) {
    if (!dir.exists(cache_path)) {
      dir.create(cache_path, recursive = TRUE)
    }
    cache_file <- file.path(cache_path, paste0(cache_id, "_order", max_order, ".rds"))
    saveRDS(interaction_features_df, cache_file)
    message(sprintf("Cached interaction features to %s", cache_file))
  }
  
  return(interaction_features_df)
}

#' 特征选择函数
#' @param X 特征矩阵
#' @param y 目标变量
#' @param method 特征选择方法 ("lasso", "correlation", "variance")
#' @param top_n 选择的特征数量
#' @return 选择的特征索引
select_features <- function(X, y, method = "lasso", top_n = 100) {
  if (method == "lasso") {
    cv_fit <- cv.glmnet(X, y, alpha = 1, nfolds = 5)
    coefs <- coef(cv_fit, s = "lambda.min")[-1]
    feature_importance <- abs(coefs)
    selected_idx <- order(feature_importance, decreasing = TRUE)[1:min(top_n, length(feature_importance))]
    
  } else if (method == "correlation") {
    cors <- apply(X, 2, function(x) abs(cor(x, y)))
    selected_idx <- order(cors, decreasing = TRUE)[1:min(top_n, length(cors))]
    
  } else if (method == "variance") {
    vars <- apply(X, 2, var)
    selected_idx <- order(vars, decreasing = TRUE)[1:min(top_n, length(vars))]
    
  } else {
    stop("Unknown feature selection method")
  }
  
  return(selected_idx)
}

#' 数据集划分函数，包括训练、验证和测试集
#' @param X 特征矩阵
#' @param y 目标变量
#' @param train_ratio 训练集比例
#' @param val_ratio 验证集比例
#' @param test_ratio 测试集比例
#' @param random_seed 随机种子
#' @return 包含划分的数据集的列表
split_dataset <- function(X, y, 
                          train_ratio = 0.7, 
                          val_ratio = 0.15, 
                          test_ratio = 0.15, 
                          random_seed = 42) {
  set.seed(random_seed)
  
  if (abs(train_ratio + val_ratio + test_ratio - 1) > 1e-10) {
    stop("Train, validation and test ratios must sum to 1")
  }
  
  n_samples <- nrow(X)
  indices <- sample(1:n_samples)
  
  n_train <- floor(n_samples * train_ratio)
  n_val <- floor(n_samples * val_ratio)
  
  train_idx <- indices[1:n_train]
  val_idx <- indices[(n_train+1):(n_train+n_val)]
  test_idx <- indices[(n_train+n_val+1):n_samples]
  
  result <- list(
    X_train = X[train_idx, , drop = FALSE],
    y_train = y[train_idx, drop = FALSE],
    X_val = X[val_idx, , drop = FALSE],
    y_val = y[val_idx, drop = FALSE],
    X_test = X[test_idx, , drop = FALSE],
    y_test = y[test_idx, drop = FALSE],
    train_idx = train_idx,
    val_idx = val_idx,
    test_idx = test_idx
  )
  
  result$sizes <- list(
    train = length(train_idx),
    val = length(val_idx),
    test = length(test_idx),
    total = n_samples
  )
  
  return(result)
}

# ======== 评估指标计算函数 ========

#' 计算回归评估指标
#' @param true 实际值
#' @param pred 预测值
#' @param metrics 要计算的指标向量
#' @return 包含评估指标的列表
calculate_metrics <- function(true, pred, 
                              metrics = c("rmse", "mae", "r2", "mape")
) {
  true <- as.numeric(true)
  pred <- as.numeric(pred)
  
  results <- list()
  
  if ("mse" %in% metrics || "rmse" %in% metrics) {
    mse <- mean((true - pred)^2, na.rm = TRUE)
    results$mse <- mse
    
    if ("rmse" %in% metrics) {
      results$rmse <- sqrt(mse)
    }
  }
  
  if ("mae" %in% metrics) {
    results$mae <- mean(abs(true - pred), na.rm = TRUE)
  }
  
  if ("r2" %in% metrics) {
    ss_res <- sum((true - pred)^2, na.rm = TRUE)
    ss_tot <- sum((true - mean(true, na.rm = TRUE))^2, na.rm = TRUE)
    results$r2 <- max(0, 1 - ss_res/ss_tot)
  }
  
  if ("rmsle" %in% metrics) {
    safe_true <- pmax(true, 1e-10)
    safe_pred <- pmax(pred, 1e-10)
    results$rmsle <- sqrt(mean((log(safe_true + 1) - log(safe_pred + 1))^2, na.rm = TRUE))
  }
  
  # 计算平均绝对百分比误差
  if ("mape" %in% metrics) {

    nonzero <- which(abs(true) > 1e-10)
    if (length(nonzero) > 0) {
      results$mape <- mean(abs((true[nonzero] - pred[nonzero]) / true[nonzero]), na.rm = TRUE) * 100
    } else {
      results$mape <- NA
    }
  }
  
  # 计算解释方差比
  if ("explained_variance" %in% metrics) {
    var_true <- var(true, na.rm = TRUE)
    var_error <- var(true - pred, na.rm = TRUE)
    results$explained_variance <- max(0, 1 - var_error/var_true)
  }
  
  return(results)
}

# ======== 模型训练和评估函数 ========

#' 数据标准化函数
#' @param data 需要标准化的数据
#' @param means 均值向量，默认为NULL
#' @param sds 标准差向量，默认为NULL
#' @return 标准化后的数据和标准化参数
standardize_data <- function(data, means = NULL, sds = NULL) {
  # 如果未提供均值和标准差，则计算
  if (is.null(means)) {
    means <- colMeans(data, na.rm = TRUE)
  }
  if (is.null(sds)) {
    sds <- apply(data, 2, sd, na.rm = TRUE)
    # 处理常数列
    sds[sds < 1e-10] <- 1
  }
  
  # 执行标准化
  scaled_data <- scale(data, center = means, scale = sds)
  
  return(list(
    data = scaled_data,
    means = means,
    sds = sds
  ))
}

#' Lasso模型训练函数
#' @param X_train 训练特征矩阵
#' @param y_train 训练目标变量
#' @param params 模型参数
#' @param X_val 验证特征矩阵
#' @param y_val 验证目标变量
#' @return 训练好的模型和相关信息
train_lasso <- function(X_train, 
                        y_train, 
                        params, 
                        X_val = NULL, 
                        y_val = NULL) {
  # 数据标准化
  scaled <- standardize_data(X_train)
  X_scaled <- scaled$data
  scaler <- list(means = scaled$means, sds = scaled$sds)
  
  # 交叉验证训练
  cv_fit <- cv.glmnet(
    x = X_scaled,
    y = y_train,
    alpha = params$alpha,
    nfolds = params$nfolds,
    standardize = FALSE,
    type.measure = "mse"
  )
  
  # 根据规则选择最佳lambda
  best_lambda <- ifelse(
    params$lambda_rule == "lambda.min",
    cv_fit$lambda.min,
    cv_fit$lambda.1se
  )
  
  # 使用最佳lambda训练最终模型
  final_model <- glmnet(
    x = X_scaled,
    y = y_train,
    alpha = params$alpha,
    lambda = best_lambda,
    standardize = FALSE
  )
  
  # 提取非零系数
  coefs <- coef(final_model)[-1]
  nonzero_idx <- which(abs(coefs) > 1e-8)
  nonzero_coefs <- coefs[nonzero_idx]
  nonzero_names <- colnames(X_train)[nonzero_idx]
  
  # 计算特征重要性
  feature_importance <- data.frame(
    feature = colnames(X_train),
    importance = abs(coef(final_model)[-1]),
    coefficient = coef(final_model)[-1],
    stringsAsFactors = FALSE
  )
  feature_importance <- feature_importance[order(feature_importance$importance, decreasing = TRUE), ]
  
  # 构建结果
  result <- list(
    model = final_model,
    cv_model = cv_fit,
    best_lambda = best_lambda,
    scaler = scaler,
    nonzero_features = list(
      indices = nonzero_idx,
      names = nonzero_names,
      coefficients = nonzero_coefs,
      count = length(nonzero_idx)
    ),
    feature_importance = feature_importance,
    model_type = "lasso"
  )
  
  if (!is.null(X_val) && !is.null(y_val)) {
    # 标准化验证集
    X_val_scaled <- scale(X_val, center = scaler$means, scale = scaler$sds)
    
    # 预测验证集
    val_pred <- predict(final_model, newx = X_val_scaled)
    
    # 计算验证指标
    val_metrics <- calculate_metrics(
      true = y_val,
      pred = val_pred,
      metrics = config$evaluation$test_eval_metrics
    )
    
    result$validation <- list(
      predictions = val_pred,
      metrics = val_metrics
    )
  }
  
  return(result)
}

#' MLP模型训练函数
#' @param X_train 训练特征矩阵
#' @param y_train 训练目标变量
#' @param params 模型参数
#' @param X_val 验证特征矩阵
#' @param y_val 验证目标变量
#' @return 训练好的模型和相关信息
train_mlp <- function(X_train, y_train, 
                      params, X_val = NULL, y_val = NULL) {
  # 数据标准化
  scaled <- standardize_data(X_train)
  X_scaled <- scaled$data
  scaler <- list(means = scaled$means, sds = scaled$sds)
  
  # 构建模型
  model <- keras_model_sequential()
  
  # 输入层
  model %>%
    layer_dense(
      units = params$layers[1],
      activation = params$activation,
      input_shape = ncol(X_scaled),
      kernel_regularizer = regularizer_l2(params$l2_reg),
      kernel_initializer = "he_normal",
      name = "input_layer"
    )
  
  # 隐藏层
  for (i in seq_along(params$layers)[-1]) {
    if (params$batch_norm) {
      model %>% layer_batch_normalization(name = paste0("batch_norm_", i))
    }
    
    model %>%
      layer_dropout(rate = params$dropout[i], name = paste0("dropout_", i)) %>%
      layer_dense(
        units = params$layers[i],
        activation = params$activation,
        kernel_regularizer = regularizer_l2(params$l2_reg),
        name = paste0("hidden_", i)
      )
  }
  
  # 输出层
  model %>% layer_dense(units = 1, activation = "linear", name = "output")
  
  # 编译模型
  model %>% compile(
    optimizer = optimizer_adam(learning_rate = params$learning_rate),
    loss = "mse",
    metrics = c("mae")
  )
  
  callbacks_list <- list()
  
  # 早停回调
  early_stop <- callback_early_stopping(
    monitor = params$early_stopping$monitor,
    patience = params$early_stopping$patience,
    restore_best_weights = params$early_stopping$restore_best_weights,
    min_delta = params$early_stopping$min_delta,
    verbose = 1
  )
  callbacks_list <- c(callbacks_list, list(early_stop))
  
  log_dir <- file.path(paths$output_dir, "logs", paste0("mlp_", format(Sys.time(), "%Y%m%d_%H%M%S")))
  if (!dir.exists(log_dir)) {
    dir.create(log_dir, recursive = TRUE)
  }
  tensorboard_callback <- callback_tensorboard(log_dir = log_dir)
  callbacks_list <- c(callbacks_list, list(tensorboard_callback))
  
  # 模型保存回调
  if (params$early_stopping$restore_best_weights) {
    checkpoint_path <- file.path(paths$model_dir, "mlp_best_weights.h5")
    checkpoint_callback <- callback_model_checkpoint(
      filepath = checkpoint_path,
      save_best_only = TRUE,
      save_weights_only = TRUE,
      monitor = params$early_stopping$monitor,
      verbose = 1
    )
    callbacks_list <- c(callbacks_list, list(checkpoint_callback))
  }
  
  # 准备验证数据
  validation_data <- NULL
  if (!is.null(X_val) && !is.null(y_val)) {
    # 标准化验证集
    X_val_scaled <- scale(X_val, center = scaler$means, scale = scaler$sds)
    validation_data <- list(X_val_scaled, y_val)
  }
  
  # 训练模型
  history <- model %>% fit(
    x = X_scaled,
    y = y_train,
    epochs = params$epochs,
    batch_size = params$batch_size,
    validation_split = ifelse(is.null(validation_data), params$validation_split, 0),
    validation_data = validation_data,
    callbacks = callbacks_list,
    verbose = 1
  )
  
  # 计算最佳轮次和模型性能
  best_epoch <- which.min(history$metrics[[params$early_stopping$monitor]])
  best_val_loss <- min(history$metrics[[params$early_stopping$monitor]])
  
  # 导出训练历史
  history_df <- data.frame(
    epoch = 1:length(history$metrics$loss),
    loss = history$metrics$loss,
    val_loss = history$metrics[[params$early_stopping$monitor]],
    stringsAsFactors = FALSE
  )
  
  # 验证集预测
  validation <- NULL
  if (!is.null(X_val) && !is.null(y_val)) {
    # 标准化验证集
    X_val_scaled <- scale(X_val, center = scaler$means, scale = scaler$sds)
    
    # 预测验证集
    val_pred <- model %>% predict(X_val_scaled)
    
    # 计算验证指标
    val_metrics <- calculate_metrics(
      true = y_val,
      pred = val_pred,
      metrics = config$evaluation$test_eval_metrics
    )
    
    validation <- list(
      predictions = val_pred,
      metrics = val_metrics
    )
  }
  
  # 构建和返回结果
  result <- list(
    model = model,
    history = history,
    history_df = history_df,
    scaler = scaler,
    best_epoch = best_epoch,
    best_val_loss = best_val_loss,
    validation = validation,
    model_type = "mlp",
    model_architecture = get_config(model)
  )
  
  return(result)
}

#' XGBoost模型训练函数
#' @param X_train 训练特征矩阵
#' @param y_train 训练目标变量 
#' @param params 模型参数
#' @param X_val 验证特征矩阵
#' @param y_val 验证目标变量
#' @return 训练好的模型和相关信息
train_xgboost <- function(X_train, y_train, params, X_val = NULL, y_val = NULL) {
  # 准备数据
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_train)
  
  # 准备验证集
  watchlist <- list(train = dtrain)
  if (!is.null(X_val) && !is.null(y_val)) {
    dval <- xgb.DMatrix(data = as.matrix(X_val), label = y_val)
    watchlist <- c(watchlist, list(val = dval))
  }
  
  # 交叉验证确定最佳迭代次数
  cv_result <- xgb.cv(
    params = params,
    data = dtrain,
    nrounds = params$nrounds,
    nfold = 5,
    stratified = FALSE,
    early_stopping_rounds = params$early_stopping_rounds,
    metrics = params$eval_metric,
    verbose = 0
  )
  
  # 确定最佳迭代次数
  best_iteration <- which.min(cv_result$evaluation_log$test_rmse_mean)
  
  # 使用最佳迭代次数训练最终模型
  final_model <- xgb.train(
    params = params,
    data = dtrain,
    nrounds = best_iteration,
    watchlist = watchlist,
    verbose = 0
  )
  
  # 计算特征重要性
  importance_matrix <- xgb.importance(
    feature_names = colnames(X_train),
    model = final_model
  )
  
  # 记录训练过程
  train_log <- data.frame(
    iteration = 1:length(final_model$evaluation_log$train_rmse),
    train_rmse = final_model$evaluation_log$train_rmse,
    stringsAsFactors = FALSE
  )
  
  # 添加验证指标
  if (!is.null(X_val) && !is.null(y_val)) {
    if ("val_rmse" %in% names(final_model$evaluation_log)) {
      train_log$val_rmse <- final_model$evaluation_log$val_rmse
    }
  }
  
  # 构建结果
  result <- list(
    model = final_model,
    cv_result = cv_result,
    best_iteration = best_iteration,
    feature_importance = importance_matrix,
    train_log = train_log,
    model_type = "xgboost"
  )
  
  # 计算验证指标
  if (!is.null(X_val) && !is.null(y_val)) {
    # 预测验证集
    val_pred <- predict(final_model, dval)
    
    # 计算验证指标
    val_metrics <- calculate_metrics(
      true = y_val,
      pred = val_pred,
      metrics = config$evaluation$test_eval_metrics
    )
    
    result$validation <- list(
      predictions = val_pred,
      metrics = val_metrics
    )
  }
  
  return(result)
}

# ======== 交叉验证函数 ========

#' 统一的交叉验证训练框架
#' @param X 特征矩阵
#' @param y 目标变量
#' @param model_name 模型名称("lasso", "mlp", "xgboost")
#' @param params 模型参数
#' @param k 交叉验证折数
#' @param random_seed 随机种子
#' @param stratified 是否进行分层抽样
#' @return 交叉验证结果列表
train_with_cv <- function(X, y, model_name, params, k = 5, random_seed = 42, stratified = FALSE) {
  # 设置随机种子
  set.seed(random_seed)
  
  # 生成交叉验证折叠
  if (stratified && is.factor(y)) {
    # 分类问题的分层抽样
    folds <- createFolds(y, k = k, list = TRUE, returnTrain = FALSE)
  } else {
    # 回归问题的随机抽样
    folds <- createFolds(y, k = k, list = TRUE, returnTrain = FALSE)
  }
  
  # 初始化结果存储
  cv_models <- vector("list", k)
  cv_metrics <- vector("list", k)
  all_predictions <- list(
    train = vector("list", k),
    val = vector("list", k)
  )
  
  # 记录开始时间
  start_time <- Sys.time()
  message(sprintf("Starting %d-fold cross-validation for %s model...", k, model_name))
  
  # 对每个折执行训练和评估
  for (fold in 1:k) {
    message(sprintf("Training fold %d/%d...", fold, k))
    
    # 获取训练集和验证集索引
    val_idx <- folds[[fold]]
    train_idx <- setdiff(1:length(y), val_idx)
    
    # 准备数据
    X_train_fold <- X[train_idx, , drop = FALSE]
    y_train_fold <- y[train_idx, drop = FALSE]
    X_val_fold <- X[val_idx, , drop = FALSE]
    y_val_fold <- y[val_idx, drop = FALSE]
    
    # 根据模型类型进行训练
    if (model_name == "lasso") {
      model <- train_lasso(X_train_fold, y_train_fold, params, X_val_fold, y_val_fold)
    } else if (model_name == "mlp") {
      model <- train_mlp(X_train_fold, y_train_fold, params, X_val_fold, y_val_fold)
    } else if (model_name == "xgboost") {
      model <- train_xgboost(X_train_fold, y_train_fold, params, X_val_fold, y_val_fold)
    } else {
      stop(sprintf("Unknown model type: %s", model_name))
    }
    
    # 保存模型
    cv_models[[fold]] <- model
    
    # 预测并计算指标
    if (model_name == "lasso") {
      # 标准化数据
      X_train_scaled <- scale(X_train_fold, center = model$scaler$means, scale = model$scaler$sds)
      X_val_scaled <- scale(X_val_fold, center = model$scaler$means, scale = model$scaler$sds)
      
      # 预测
      train_pred <- predict(model$model, newx = X_train_scaled)
      val_pred <- predict(model$model, newx = X_val_scaled)
    } else if (model_name == "mlp") {
      # 标准化数据
      X_train_scaled <- scale(X_train_fold, center = model$scaler$means, scale = model$scaler$sds)
      X_val_scaled <- scale(X_val_fold, center = model$scaler$means, scale = model$scaler$sds)
      
      # 预测
      train_pred <- model$model %>% predict(X_train_scaled)
      val_pred <- model$model %>% predict(X_val_scaled)
    } else if (model_name == "xgboost") {
      # XGBoost不需要标准化
      dtrain <- xgb.DMatrix(data = as.matrix(X_train_fold), label = y_train_fold)
      dval <- xgb.DMatrix(data = as.matrix(X_val_fold), label = y_val_fold)
      
      # 预测
      train_pred <- predict(model$model, dtrain)
      val_pred <- predict(model$model, dval)
    }
    
    # 存储预测值
    all_predictions$train[[fold]] <- data.frame(
      fold = fold,
      idx = train_idx,
      actual = y_train_fold,
      predicted = train_pred
    )
    
    all_predictions$val[[fold]] <- data.frame(
      fold = fold,
      idx = val_idx,
      actual = y_val_fold,
      predicted = val_pred
    )
    
    # 计算评估指标
    train_metrics <- calculate_metrics(y_train_fold, train_pred, config$evaluation$test_eval_metrics)
    val_metrics <- calculate_metrics(y_val_fold, val_pred, config$evaluation$test_eval_metrics)
    
    # 存储评估指标
    cv_metrics[[fold]] <- list(
      train = train_metrics,
      val = val_metrics
    )
    
    # 显示当前折的评估结果
    message(sprintf("  Fold %d - Train RMSE: %.4f, Val RMSE: %.4f, Train R²: %.4f, Val R²: %.4f",
                    fold, train_metrics$rmse, val_metrics$rmse, 
                    train_metrics$r2, val_metrics$r2))
  }
  
  # 计算平均评估指标
  avg_train_metrics <- list()
  avg_val_metrics <- list()
  
  # 初始化指标列表
  for (metric in config$evaluation$test_eval_metrics) {
    avg_train_metrics[[metric]] <- mean(sapply(cv_metrics, function(x) x$train[[metric]]))
    avg_val_metrics[[metric]] <- mean(sapply(cv_metrics, function(x) x$val[[metric]]))
  }
  
  # 计算指标标准差
  sd_train_metrics <- list()
  sd_val_metrics <- list()
  
  for (metric in config$evaluation$test_eval_metrics) {
    sd_train_metrics[[metric]] <- sd(sapply(cv_metrics, function(x) x$train[[metric]]))
    sd_val_metrics[[metric]] <- sd(sapply(cv_metrics, function(x) x$val[[metric]]))
  }
  
  # 合并预测结果
  combined_train_predictions <- do.call(rbind, all_predictions$train)
  combined_val_predictions <- do.call(rbind, all_predictions$val)
  
  # 所有样本的预测
  all_sample_predictions <- rbind(
    combined_train_predictions,
    combined_val_predictions
  )
  all_sample_predictions <- all_sample_predictions[order(all_sample_predictions$idx), ]
  
  # 记录结束时间
  end_time <- Sys.time()
  elapsed <- difftime(end_time, start_time, units = "mins")
  message(sprintf("Cross-validation completed in %.2f minutes.", as.numeric(elapsed)))
  message(sprintf("Average validation RMSE: %.4f (±%.4f)", 
                  avg_val_metrics$rmse, sd_val_metrics$rmse))
  message(sprintf("Average validation R²: %.4f (±%.4f)", 
                  avg_val_metrics$r2, sd_val_metrics$r2))
  
  # 返回完整结果
  result <- list(
    models = cv_models,
    metrics = list(
      by_fold = cv_metrics,
      avg_train = avg_train_metrics,
      avg_val = avg_val_metrics,
      sd_train = sd_train_metrics,
      sd_val = sd_val_metrics
    ),
    predictions = list(
      train = combined_train_predictions,
      val = combined_val_predictions,
      all = all_sample_predictions
    ),
    folds = folds,
    runtime = as.numeric(elapsed),
    model_name = model_name,
    params = params,
    n_folds = k
  )
  
  return(result)
}

# ======== 测试集评估函数 ========

#' 在测试集上评估模型
#' @param model 训练好的模型
#' @param X_test 测试特征矩阵
#' @param y_test 测试目标变量
#' @param model_name 模型名称
#' @return 测试结果列表
evaluate_on_test <- function(model, X_test, y_test, model_name) {
  # 根据模型类型进行预测
  if (model_name == "lasso") {
    # 标准化测试数据
    X_test_scaled <- scale(X_test, center = model$scaler$means, scale = model$scaler$sds)
    # 预测
    y_pred <- predict(model$model, newx = X_test_scaled)
  } else if (model_name == "mlp") {
    # 标准化测试数据
    X_test_scaled <- scale(X_test, center = model$scaler$means, scale = model$scaler$sds)
    # 预测
    y_pred <- model$model %>% predict(X_test_scaled)
  } else if (model_name == "xgboost") {
    # XGBoost无需标准化
    dtest <- xgb.DMatrix(data = as.matrix(X_test))
    # 预测
    y_pred <- predict(model$model, dtest)
  } else {
    stop(sprintf("Unknown model type: %s", model_name))
  }
  
  # 计算评估指标
  metrics <- calculate_metrics(y_test, y_pred, config$evaluation$test_eval_metrics)
  
  # 构建预测结果数据框
  predictions_df <- data.frame(
    actual = y_test,
    predicted = y_pred,
    error = y_test - y_pred,
    abs_error = abs(y_test - y_pred),
    stringsAsFactors = FALSE
  )
  
  # 计算预测统计信息
  pred_stats <- list(
    mean_error = mean(predictions_df$error),
    median_error = median(predictions_df$error),
    mean_abs_error = mean(predictions_df$abs_error),
    median_abs_error = median(predictions_df$abs_error),
    std_error = sd(predictions_df$error),
    min_error = min(predictions_df$error),
    max_error = max(predictions_df$error),
    q1_error = quantile(predictions_df$error, 0.25),
    q3_error = quantile(predictions_df$error, 0.75)
  )
  
  # 返回测试结果
  result <- list(
    metrics = metrics,
    predictions = predictions_df,
    prediction_stats = pred_stats,
    model_name = model_name
  )
  
  return(result)
}

# ======== 集成模型函数 ========

#' 创建并评估简单平均集成模型
#' @param models 模型列表
#' @param X_test 测试特征矩阵
#' @param y_test 测试目标变量
#' @param weights 各模型权重，默认为NULL表示平均
#' @return 集成模型评估结果
create_ensemble <- function(models, X_test, y_test, weights = NULL) {
  # 获取模型数量
  n_models <- length(models)
  
  # 如果未提供权重，使用平均权重
  if (is.null(weights)) {
    weights <- rep(1/n_models, n_models)
  } else if (length(weights) != n_models) {
    stop("Number of weights must match number of models")
  }
  
  # 获取各模型的预测
  predictions <- vector("list", n_models)
  model_names <- vector("character", n_models)
  
  for (i in 1:n_models) {
    model <- models[[i]]
    model_name <- model$model_type
    model_names[i] <- model_name
    
    # 根据模型类型进行预测
    if (model_name == "lasso") {
      X_test_scaled <- scale(X_test, center = model$scaler$means, scale = model$scaler$sds)
      predictions[[i]] <- predict(model$model, newx = X_test_scaled)
    } else if (model_name == "mlp") {
      X_test_scaled <- scale(X_test, center = model$scaler$means, scale = model$scaler$sds)
      predictions[[i]] <- model$model %>% predict(X_test_scaled)
    } else if (model_name == "xgboost") {
      dtest <- xgb.DMatrix(data = as.matrix(X_test))
      predictions[[i]] <- predict(model$model, dtest)
    }
  }
  
  # 计算加权平均预测
  ensemble_pred <- rep(0, length(y_test))
  for (i in 1:n_models) {
    ensemble_pred <- ensemble_pred + weights[i] * predictions[[i]]
  }
  
  # 计算集成模型评估指标
  ensemble_metrics <- calculate_metrics(y_test, ensemble_pred, config$evaluation$test_eval_metrics)
  
  # 构建各模型及集成模型的预测结果数据框
  predictions_df <- data.frame(
    actual = y_test,
    ensemble = ensemble_pred,
    stringsAsFactors = FALSE
  )
  
  # 添加各模型的预测结果
  for (i in 1:n_models) {
    predictions_df[[model_names[i]]] <- predictions[[i]]
  }
  
  # 计算各模型与实际值的相关性
  correlations <- sapply(model_names, function(name) {
    cor(predictions_df$actual, predictions_df[[name]])
  })
  correlations <- c(correlations, ensemble = cor(predictions_df$actual, predictions_df$ensemble))
  
  # 返回集成模型结果
  result <- list(
    metrics = ensemble_metrics,
    predictions = predictions_df,
    weights = weights,
    model_names = model_names,
    correlations = correlations
  )
  
  return(result)
}


# ======== 数据可视化函数 ========

#' 创建特征重要性条形图
#' @param importance_df 特征重要性数据框
#' @param top_n 显示前N个特征
#' @param title 图表标题
#' @param x_label x轴标签
#' @param y_label y轴标签
#' @param color 条形颜色
#' @param filename 输出文件名
#' @return ggplot对象
plot_feature_importance <- function(importance_df, top_n = 15, 
                                    title = "Feature Importance", 
                                    x_label = "Importance", 
                                    y_label = "Feature",
                                    color = config$plotting$importance_color,
                                    filename = NULL) {
  # 确保输入是数据框并按重要性排序
  if (!is.data.frame(importance_df)) {
    importance_df <- as.data.frame(importance_df)
  }
  
  # 确保数据框有feature和importance列
  required_cols <- c("feature", "importance")
  if (!all(required_cols %in% colnames(importance_df))) {
    stop("importance_df must have 'feature' and 'importance' columns")
  }
  
  # 按重要性排序并取前N个
  importance_df <- importance_df[order(importance_df$importance, decreasing = TRUE), ]
  if (nrow(importance_df) > top_n) {
    importance_df <- importance_df[1:top_n, ]
  }
  
  # 将特征名称转换为因子，保持排序
  importance_df$feature <- factor(importance_df$feature, 
                                  levels = rev(importance_df$feature))
  
  # 创建条形图
  p <- ggplot(importance_df, aes(x = importance, y = feature)) +
    geom_bar(stat = "identity", fill = color) +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      axis.text.y = element_text(hjust = 1),
      panel.grid.major.y = element_blank(),
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 20, 20, 20)
    )
  
  # 添加颜色编码
  if ("coefficient" %in% colnames(importance_df)) {
    p <- ggplot(importance_df, aes(x = importance, y = feature, 
                                   fill = coefficient > 0)) +
      geom_bar(stat = "identity") +
      scale_fill_manual(values = c("firebrick3", "steelblue3"), 
                        name = "Effect",
                        labels = c("Negative", "Positive")) +
      labs(title = title, x = x_label, y = y_label) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
        axis.title = element_text(size = 12),
        axis.text = element_text(size = 10),
        axis.text.y = element_text(hjust = 1),
        panel.grid.major.y = element_blank(),
        panel.grid.minor = element_blank(),
        plot.margin = margin(20, 20, 20, 20),
        legend.position = "bottom"
      )
  }
  
  # 保存图表
  if (!is.null(filename)) {
    save_path <- file.path(paths$plot_dir, paste0(filename, ".", config$plotting$save_format))
    ggsave(
      filename = save_path,
      plot = p,
      width = config$plotting$width,
      height = config$plotting$height,
      dpi = config$plotting$dpi
    )
    message(sprintf("Feature importance plot saved to: %s", save_path))
  }
  
  return(p)
}

#' 创建实际值vs预测值散点图
#' @param predictions_df 包含actual和predicted列的数据框
#' @param title 图表标题
#' @param x_label x轴标签
#' @param y_label y轴标签
#' @param color 点颜色
#' @param filename 输出文件名
#' @return ggplot对象
plot_actual_vs_predicted <- function(predictions_df, 
                                     title = "Actual vs. Predicted Values",
                                     x_label = "Actual Values",
                                     y_label = "Predicted Values",
                                     color = config$plotting$point_color,
                                     filename = NULL) {
  # 确保数据框有actual和predicted列
  required_cols <- c("actual", "predicted")
  if (!all(required_cols %in% colnames(predictions_df))) {
    stop("predictions_df must have 'actual' and 'predicted' columns")
  }
  
  # 计算最小值和最大值以创建对角线
  min_val <- min(c(predictions_df$actual, predictions_df$predicted), na.rm = TRUE)
  max_val <- max(c(predictions_df$actual, predictions_df$predicted), na.rm = TRUE)
  range_vals <- data.frame(x = c(min_val, max_val), y = c(min_val, max_val))
  
  # 计算R²和RMSE
  r2 <- with(predictions_df, 1 - sum((actual - predicted)^2) / sum((actual - mean(actual))^2))
  rmse <- with(predictions_df, sqrt(mean((actual - predicted)^2)))
  
  # 创建散点图
  p <- ggplot(predictions_df, aes(x = actual, y = predicted)) +
    geom_point(color = color, alpha = 0.7) +
    geom_line(data = range_vals, aes(x = x, y = y), 
              linetype = "dashed", color = "gray50") +
    geom_smooth(method = "lm", color = "black", se = FALSE) +
    annotate("text", x = min_val + 0.2 * (max_val - min_val), 
             y = max_val - 0.1 * (max_val - min_val),
             label = sprintf("R² = %.4f\nRMSE = %.4f", r2, rmse),
             hjust = 0, size = 4) +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 20, 20, 20)
    ) +
    coord_fixed(ratio = 1, xlim = c(min_val, max_val), ylim = c(min_val, max_val))
  
  # 保存图表
  if (!is.null(filename)) {
    save_path <- file.path(paths$plot_dir, paste0(filename, ".", config$plotting$save_format))
    ggsave(
      filename = save_path,
      plot = p,
      width = config$plotting$width,
      height = config$plotting$height,
      dpi = config$plotting$dpi
    )
    message(sprintf("Actual vs predicted plot saved to: %s", save_path))
  }
  
  return(p)
}

#' 创建残差图
#' @param predictions_df 包含actual和predicted列的数据框
#' @param title 图表标题
#' @param x_label x轴标签
#' @param y_label y轴标签
#' @param color 点颜色
#' @param filename 输出文件名
#' @return ggplot对象
plot_residuals <- function(predictions_df,
                           title = "Residual Plot",
                           x_label = "Predicted Values",
                           y_label = "Residuals",
                           color = config$plotting$point_color,
                           filename = NULL) {
  # 确保数据框有actual和predicted列
  required_cols <- c("actual", "predicted")
  if (!all(required_cols %in% colnames(predictions_df))) {
    stop("predictions_df must have 'actual' and 'predicted' columns")
  }
  
  # 计算残差
  predictions_df$residuals <- predictions_df$actual - predictions_df$predicted
  
  # 创建残差图
  p <- ggplot(predictions_df, aes(x = predicted, y = residuals)) +
    geom_point(color = color, alpha = 0.7) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
    geom_smooth(method = "loess", color = "black", se = TRUE) +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 20, 20, 20)
    )
  
  # 保存图表
  if (!is.null(filename)) {
    save_path <- file.path(paths$plot_dir, paste0(filename, ".", config$plotting$save_format))
    ggsave(
      filename = save_path,
      plot = p,
      width = config$plotting$width,
      height = config$plotting$height,
      dpi = config$plotting$dpi
    )
    message(sprintf("Residual plot saved to: %s", save_path))
  }
  
  return(p)
}

#' 创建训练历史图表（用于MLP和XGBoost）
#' @param history_df 训练历史数据框
#' @param title 图表标题
#' @param x_label x轴标签
#' @param y_label y轴标签
#' @param filename 输出文件名
#' @return ggplot对象
plot_training_history <- function(history_df,
                                  title = "Training History",
                                  x_label = "Epoch",
                                  y_label = "Loss",
                                  metric_labels = NULL,  # 新增参数
                                  filename = NULL) {
  # 检查训练历史数据框的格式
  required_cols <- c("epoch")
  metric_cols <- setdiff(colnames(history_df), "epoch")
  
  if (!all(required_cols %in% colnames(history_df)) || length(metric_cols) == 0) {
    stop("history_df must have 'epoch' column and at least one metric column")
  }
  
  # 将数据转换为长格式
  history_long <- reshape2::melt(history_df, id.vars = "epoch", 
                                 variable.name = "metric", value.name = "value")
  
  # 创建训练历史图
  p <- ggplot(history_long, aes(x = epoch, y = value, color = metric)) +
    geom_line(linewidth = 1) +
    labs(title = title, x = x_label, y = y_label) +
    theme_minimal() +
    theme(
      plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
      axis.title = element_text(size = 12),
      axis.text = element_text(size = 10),
      panel.grid.minor = element_blank(),
      plot.margin = margin(20, 20, 20, 20),
      legend.position = "bottom",
      legend.title = element_blank()
    ) +
    scale_color_brewer(palette = "Set1")
  
  # 自定义图例标签
  if (!is.null(metric_labels)) {
    p <- p + scale_color_discrete(labels = metric_labels)
  }
  
  # 保存图表
  if (!is.null(filename)) {
    save_path <- file.path(paths$plot_dir, paste0(filename, ".", config$plotting$save_format))
    ggsave(
      filename = save_path,
      plot = p,
      width = config$plotting$width,
      height = config$plotting$height,
      dpi = config$plotting$dpi
    )
    message(sprintf("Training history plot saved to: %s", save_path))
  }
  
  return(p)
}

#' 创建特征分布图
#' @param data 数据框
#' @param feature_name 特征名称
#' @param target 目标变量名称
#' @param bins 直方图分箱数
#' @param title 图表标题
#' @param x_label x轴标签
#' @param y_label y轴标签
#' @param filename 输出文件名
#' @return ggplot对象
plot_feature_distribution <- function(data,
                                      feature_name,
                                      target = NULL,
                                      bins = 30,
                                      title = NULL,
                                      x_label = NULL,
                                      y_label = "Frequency",
                                      filename = NULL) {
  # 检查输入
  if (!is.data.frame(data)) {
    stop("data must be a data frame")
  }
  
  if (!feature_name %in% colnames(data)) {
    stop(sprintf("Feature '%s' not found in the data", feature_name))
  }
  
  # 设置默认标题和x轴标签
  if (is.null(title)) {
    title <- sprintf("Distribution of %s", feature_name)
  }
  
  if (is.null(x_label)) {
    x_label <- feature_name
  }
  
  # 检查特征类型和创建相应图表
  if (is.numeric(data[[feature_name]])) {
    # 数值型特征：创建直方图或密度图
    if (is.null(target)) {
      # 没有目标变量：创建简单直方图
      p <- ggplot(data, aes(x = .data[[feature_name]])) +
        geom_histogram(bins = bins, fill = "steelblue", color = "white", alpha = 0.7) +
        geom_density(alpha = 0.2, fill = "steelblue") +
        labs(title = title, x = x_label, y = y_label) +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10),
          panel.grid.minor = element_blank(),
          plot.margin = margin(20, 20, 20, 20)
        )
    } else {
      # 有目标变量：按目标分组
      if (!target %in% colnames(data)) {
        stop(sprintf("Target '%s' not found in the data", target))
      }
      
      # 检查目标变量类型
      if (is.numeric(data[[target]]) && length(unique(data[[target]])) > 10) {
        # 数值型目标：创建散点图
        p <- ggplot(data, aes(x = .data[[feature_name]], y = .data[[target]])) +
          geom_point(alpha = 0.5, color = "steelblue") +
          geom_smooth(method = "loess", color = "firebrick") +
          labs(title = title, x = x_label, y = target) +
          theme_minimal() +
          theme(
            plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 10),
            panel.grid.minor = element_blank(),
            plot.margin = margin(20, 20, 20, 20)
          )
      } else {
        # 分类目标：创建分组密度图
        p <- ggplot(data, aes(x = .data[[feature_name]], fill = as.factor(.data[[target]]))) +
          geom_density(alpha = 0.5) +
          labs(title = title, x = x_label, y = "Density", fill = target) +
          theme_minimal() +
          theme(
            plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
            axis.title = element_text(size = 12),
            axis.text = element_text(size = 10),
            panel.grid.minor = element_blank(),
            plot.margin = margin(20, 20, 20, 20),
            legend.position = "bottom"
          ) +
          scale_fill_brewer(palette = "Set1")
      }
    }
  } else {
    # 分类特征：创建条形图
    if (is.null(target)) {
      # 没有目标变量：创建简单条形图
      # 计算频率
      freq_data <- data.frame(table(data[[feature_name]]))
      colnames(freq_data) <- c("category", "count")
      freq_data$percentage <- freq_data$count / sum(freq_data$count) * 100
      
      # 排序
      freq_data <- freq_data[order(freq_data$count, decreasing = TRUE), ]
      freq_data$category <- factor(freq_data$category, levels = freq_data$category)
      
      p <- ggplot(freq_data, aes(x = category, y = count)) +
        geom_bar(stat = "identity", fill = "steelblue", alpha = 0.7) +
        geom_text(aes(label = sprintf("%.1f%%", percentage)), 
                  vjust = -0.5, size = 3.5) +
        labs(title = title, x = x_label, y = y_label) +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10),
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.minor = element_blank(),
          plot.margin = margin(20, 20, 20, 20)
        )
    } else {
      # 有目标变量：创建分组条形图
      if (!target %in% colnames(data)) {
        stop(sprintf("Target '%s' not found in the data", target))
      }
      
      # 创建交叉表
      cross_tab <- table(data[[feature_name]], data[[target]])
      cross_df <- as.data.frame(cross_tab)
      colnames(cross_df) <- c("category", "target", "count")
      
      p <- ggplot(cross_df, aes(x = category, y = count, fill = target)) +
        geom_bar(stat = "identity", position = "dodge") +
        labs(title = title, x = x_label, y = y_label, fill = target) +
        theme_minimal() +
        theme(
          plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          axis.title = element_text(size = 12),
          axis.text = element_text(size = 10),
          axis.text.x = element_text(angle = 45, hjust = 1),
          panel.grid.minor = element_blank(),
          plot.margin = margin(20, 20, 20, 20),
          legend.position = "bottom"
        ) +
        scale_fill_brewer(palette = "Set1")
    }
  }
  
  # 保存图表
  if (!is.null(filename)) {
    save_path <- file.path(paths$plot_dir, paste0(filename, ".", config$plotting$save_format))
    ggsave(
      filename = save_path,
      plot = p,
      width = config$plotting$width,
      height = config$plotting$height,
      dpi = config$plotting$dpi
    )
    message(sprintf("Feature distribution plot saved to: %s", save_path))
  }
  
  return(p)
}

# ======== 穷举预测 ========

#' 高效生成所有可能的特征组合(1阶的，还需要再二阶交互)
#' @param feature_names 特征名称向量
#' @param max_length 最大活跃特征数量（值为1的特征数量上限），默认为全部特征数
generate_feature_combinations <- function(feature_names, 
                                          max_length = NULL) {
  # 获取特征总数
  n_features <- length(feature_names)
  
  # 如果未指定max_length，则使用所有可能的长度（即特征总数）
  if (is.null(max_length)) {
    max_length <- n_features
  } else {
    # 确保max_length不超过特征总数
    max_length <- min(max_length, n_features)
  }
  
  # 使用二进制方法直接生成所有组合
  result_list <- list()
  
  # 对于每个可能的组合长度
  for (length_i in 1:max_length) {
    # 使用combn生成特定长度的所有组合，但使用向量化操作处理
    # 生成指定长度的所有组合
    indices <- combn(n_features, length_i, simplify = FALSE)
    
    # 预分配矩阵空间以提高效率
    combo_matrix <- matrix(-1, nrow = length(indices), ncol = n_features)
    
    # 将每个组合填充到矩阵中
    for (j in 1:length(indices)) {
      combo_matrix[j, indices[[j]]] <- 1
    }
    
    colnames(combo_matrix) <- feature_names
    result_list[[length(result_list) + 1]] <- combo_matrix
  }
  
  # 合并所有结果
  result <- do.call(rbind, result_list)
  colnames(result) <- feature_names
  
  return(result)
}

# 单独使用Lasso模型预测
lasso_predict <- function(new_data, model) {
  cat("执行Lasso预测...\n")
  
  # 确保数据列名与模型特征匹配
  if (ncol(new_data) == length(model$scaler$means)) {
    colnames(new_data) <- names(model$scaler$means)
  }
  
  # 标准化
  scaled_data <- scale(new_data, 
                       center = model$scaler$means, 
                       scale = model$scaler$sds)
  
  # 预测
  predictions <- predict(model$model, newx = as.matrix(scaled_data))
  
  # 创建结果数据框
  result <- data.frame(new_data)
  result$lasso_pred <- as.numeric(predictions)
  
  # 保存结果
  write.csv(result, file.path(paths$results, "lasso_predictions.csv"))
  cat("Lasso预测结果已保存\n")
  
  # 保存排序结果
  sorted_result <- result[order(result$lasso_pred, decreasing = TRUE), ]
  write.csv(sorted_result, file.path(paths$results, "lasso_predictions_sorted.csv"))
  cat("排序后的Lasso预测结果已保存\n")
  
  return(invisible(result))
}

# 单独使用MLP模型预测
mlp_predict <- function(new_data, model) {
  cat("执行MLP预测...\n")
  
  # 确保数据列名与模型特征匹配
  if (ncol(new_data) == length(model$scaler$means)) {
    colnames(new_data) <- names(model$scaler$means)
  }
  
  # 标准化
  scaled_data <- scale(new_data, 
                       center = model$scaler$means, 
                       scale = model$scaler$sds)
  
  # 预测
  predictions <- model$model %>% predict(scaled_data)
  
  # 创建结果数据框
  result <- data.frame(new_data)
  result$mlp_pred <- as.numeric(predictions)
  
  # 保存结果
  write.csv(result, file.path(paths$results, "mlp_predictions.csv"))
  cat("MLP预测结果已保存\n")
  
  # 保存排序结果
  sorted_result <- result[order(result$mlp_pred, decreasing = TRUE), ]
  write.csv(sorted_result, file.path(paths$results, "mlp_predictions_sorted.csv"))
  cat("排序后的MLP预测结果已保存\n")
  
  return(invisible(result))
}


# 单独使用XGBoost模型预测
xgb_predict <- function(new_data, model) {
  cat("执行XGBoost预测...\n")
  
  # XGBoost不依赖特征名，只需确保数据格式正确
  dtest <- xgb.DMatrix(data = as.matrix(new_data))
  
  # 预测
  predictions <- predict(model$model, dtest)
  
  # 创建结果数据框
  result <- data.frame(new_data)
  result$xgb_pred <- as.numeric(predictions)
  
  # 保存结果
  write.csv(result, file.path(paths$results, "xgb_predictions.csv"))
  cat("XGBoost预测结果已保存\n")
  
  # 保存排序结果
  sorted_result <- result[order(result$xgb_pred, decreasing = TRUE), ]
  write.csv(sorted_result, file.path(paths$results, "xgb_predictions_sorted.csv"))
  cat("排序后的XGBoost预测结果已保存\n")
  
  return(invisible(result))
}

# 将特征值转换为特征组合描述
create_feature_combination_label <- function(row, feature_names) {
  # 找出值为1的特征
  positive_features <- character(0)
  for (i in 1:length(feature_names)) {
    if (row[feature_names[i]] == 1) {
      positive_features <- c(positive_features, feature_names[i])
    }
  }
  
  # 如果没有正值特征，返回"无"或某种默认标记
  if (length(positive_features) == 0) {
    return("无正值特征")
  }
  
  # 返回逗号连接的正值特征名
  return(paste(positive_features, collapse=", "))
}

# 1. 各模型预测值的分布直方图
model_distributions <- function(data, config) {
  # 准备数据
  dist_data <- data %>%
    select(lasso_pred, mlp_pred, xgb_pred) %>%
    pivot_longer(cols = everything(), 
                 names_to = "model", 
                 values_to = "prediction")
  
  # 创建图形
  p <- ggplot(dist_data, aes(x = prediction, fill = model)) +
    geom_histogram(alpha = 0.7, bins = 30, position = "identity") +
    scale_fill_viridis(discrete = TRUE, option = config$plotting$color_palette) +
    theme_minimal() +
    labs(title = "各模型预测值分布",
         x = "预测值",
         y = "频数",
         fill = "模型") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank())
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "model_distributions.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 2. 预测提升度对比图
improvement_comparison <- function(data, config) {
  # 准备数据
  imp_data <- data %>%
    select(lasso_improvement_pct, mlp_improvement_pct, xgb_improvement_pct) %>%
    pivot_longer(cols = everything(), 
                 names_to = "model", 
                 values_to = "improvement") %>%
    mutate(model = gsub("_improvement_pct", "", model))
  
  # 创建图形
  p <- ggplot(imp_data, aes(x = model, y = improvement, fill = model)) +
    geom_boxplot() +
    scale_fill_viridis(discrete = TRUE, option = config$plotting$color_palette) +
    theme_minimal() +
    labs(title = "各模型预测提升度对比",
         x = "模型",
         y = "预测提升度 (%)",
         fill = "模型") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank())
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "improvement_comparison.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 3. 模型预测散点图矩阵
prediction_scatterplot_matrix <- function(data, config) {
  # 选择数据
  plot_data <- data %>%
    select(lasso_pred, mlp_pred, xgb_pred)
  
  # 创建散点图函数
  plot_scatter <- function(x, y, xlab, ylab) {
    ggplot(data = data.frame(x = x, y = y), aes(x = x, y = y)) +
      geom_point(color = config$plotting$point_color, alpha = 0.5) +
      geom_smooth(method = "lm", color = config$plotting$line_color, se = FALSE) +
      theme_minimal() +
      labs(x = xlab, y = ylab) +
      theme(text = element_text(family = config$plotting$font_family),
            panel.grid = element_blank())
  }
  
  # 创建散点图矩阵
  p1 <- plot_scatter(plot_data$lasso_pred, plot_data$mlp_pred, "Lasso预测值", "MLP预测值")
  p2 <- plot_scatter(plot_data$lasso_pred, plot_data$xgb_pred, "Lasso预测值", "XGBoost预测值")
  p3 <- plot_scatter(plot_data$mlp_pred, plot_data$xgb_pred, "MLP预测值", "XGBoost预测值")
  
  # 组合图形
  p <- gridExtra::grid.arrange(p1, p2, p3, ncol = 2,
                               top = "模型预测值相关性矩阵")
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "prediction_scatterplot_matrix.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 4. 前10个最佳组合条形图
top_combinations_chart <- function(data, config) {
  # 按平均预测值计算前10个组合
  top_data <- data %>%
    mutate(avg_pred = (lasso_pred + mlp_pred + xgb_pred) / 3) %>%
    arrange(desc(avg_pred)) %>%
    head(10) %>%
    select(feature_combo, lasso_pred, mlp_pred, xgb_pred) %>%
    pivot_longer(cols = c(lasso_pred, mlp_pred, xgb_pred),
                 names_to = "model",
                 values_to = "prediction") %>%
    mutate(model = gsub("_pred", "", model),
           # 缩短特征组合标签以便显示
           feature_combo = substr(feature_combo, 1, 20),
           # 确保按平均预测值排序
           feature_combo = factor(feature_combo, 
                                  levels = rev(unique(substr(data$feature_combo[order(data$lasso_pred + data$mlp_pred + data$xgb_pred, decreasing = TRUE)], 1, 20))[1:10])))
  
  # 创建图形
  p <- ggplot(top_data, aes(x = feature_combo, y = prediction, fill = model)) +
    geom_bar(stat = "identity", position = position_dodge()) +
    scale_fill_viridis(discrete = TRUE, option = config$plotting$color_palette) +
    coord_flip() +
    theme_minimal() +
    labs(title = "三个模型预测的前10个最佳组合",
         x = "特征组合",
         y = "预测值",
         fill = "模型") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank())
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "top_combinations_chart.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 5. 各特征对预测值的影响热图
feature_impact_heatmap <- function(data, config) {
  # 准备数据
  feature_cols <- raw_data_result$feature_names
  
  # 计算每个特征对各模型预测值的相关性
  corr_data <- data.frame(feature = character(),
                          model = character(),
                          correlation = numeric())
  
  for (feat in feature_cols) {
    for (model in c("lasso_pred", "mlp_pred", "xgb_pred")) {
      # 计算相关系数
      corr <- cor(data[[feat]], data[[model]], method = "pearson")
      corr_data <- rbind(corr_data, 
                         data.frame(feature = feat,
                                    model = gsub("_pred", "", model),
                                    correlation = corr))
    }
  }
  
  # 创建热图
  p <- ggplot(corr_data, aes(x = model, y = feature, fill = correlation)) +
    geom_tile() +
    scale_fill_viridis(option = config$plotting$color_palette) +
    theme_minimal() +
    labs(title = "特征对模型预测值的相关性热图",
         x = "模型",
         y = "特征",
         fill = "相关系数") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank())
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "feature_impact_heatmap.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 6. 预测提升度范围分布图
improvement_range_chart <- function(data, config) {
  # 计算各区间的频数
  bin_size <- 5  # 每5%一个区间
  
  improvement_bins <- data %>%
    select(lasso_improvement_pct, mlp_improvement_pct, xgb_improvement_pct) %>%
    pivot_longer(cols = everything(), 
                 names_to = "model", 
                 values_to = "improvement") %>%
    mutate(model = gsub("_improvement_pct", "", model),
           bin = cut(improvement, 
                     breaks = seq(-50, 50, by = bin_size),
                     labels = paste0(seq(-50, 45, by = bin_size), "% - ", seq(-45, 50, by = bin_size), "%"),
                     include.lowest = TRUE))
  
  # 创建图形
  p <- ggplot(improvement_bins, aes(x = bin, fill = model)) +
    geom_bar(position = "dodge") +
    scale_fill_viridis(discrete = TRUE, option = config$plotting$color_palette) +
    theme_minimal() +
    labs(title = "各模型预测提升度分布",
         x = "提升度区间",
         y = "组合数量",
         fill = "模型") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank(),
          axis.text.x = element_text(angle = 45, hjust = 1))
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "improvement_range_chart.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 7. 特征组合规模(特征个数)对预测值的影响
feature_count_impact <- function(data, config) {
  # 计算每个组合包含的特征数量
  data$feature_count <- rowSums(data[, raw_data_result$feature_names])
  
  # 准备数据
  count_data <- data %>%
    select(feature_count, lasso_pred, mlp_pred, xgb_pred) %>%
    pivot_longer(cols = c(lasso_pred, mlp_pred, xgb_pred),
                 names_to = "model",
                 values_to = "prediction") %>%
    mutate(model = gsub("_pred", "", model))
  
  # 创建图形
  p <- ggplot(count_data, aes(x = as.factor(feature_count), y = prediction, fill = model)) +
    geom_boxplot() +
    scale_fill_viridis(discrete = TRUE, option = config$plotting$color_palette) +
    theme_minimal() +
    labs(title = "特征数量对预测值的影响",
         x = "特征数量",
         y = "预测值",
         fill = "模型") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank())
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "feature_count_impact.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# 8. 模型一致性分析图
model_agreement_chart <- function(data, config) {
  # 计算模型间的一致性
  data <- data %>%
    mutate(
      # 标准化预测值以便比较
      lasso_rank = percent_rank(lasso_pred),
      mlp_rank = percent_rank(mlp_pred),
      xgb_rank = percent_rank(xgb_pred),
      # 计算模型预测排名的平均差异(越小表示越一致)
      agreement_score = 1 - (abs(lasso_rank - mlp_rank) + 
                               abs(lasso_rank - xgb_rank) + 
                               abs(mlp_rank - xgb_rank)) / 3,
      # 计算平均预测值
      avg_pred = (lasso_pred + mlp_pred + xgb_pred) / 3
    )
  
  # 选择前100个平均预测值最高的组合
  top_data <- data %>%
    arrange(desc(avg_pred)) %>%
    head(100)
  
  # 创建散点图 - x轴为平均预测值，y轴为一致性得分，颜色为特征数量
  p <- ggplot(top_data, aes(x = avg_pred, y = agreement_score, color = rowSums(top_data[, raw_data_result$feature_names]))) +
    geom_point(size = 3, alpha = 0.7) +
    scale_color_viridis(option = config$plotting$color_palette) +
    theme_minimal() +
    labs(title = "模型预测一致性分析(前100个组合)",
         x = "平均预测值",
         y = "模型一致性得分(越高越一致)",
         color = "特征数量") +
    theme(text = element_text(family = config$plotting$font_family),
          panel.grid = element_blank())
  
  # 保存图形
  ggsave(file.path(features_plots_dir, "model_agreement_chart.png"),
         plot = p,
         width = config$plotting$width,
         height = config$plotting$height,
         dpi = config$plotting$dpi)
  
  return(p)
}

# -------- 创建预测值的互作网络可视化 --------
# 将预测数据转换为互作网络边的格式
prepare_feature_interaction_data <- function(pred_data, model_name, feature_cols, threshold_diff = NULL) {
  # 创建一个空的数据框来存储互作结果
  results <- data.frame(
    Source = character(),
    Target = character(),
    Difference = numeric(),
    Interaction = character(),
    Metric_Diff = numeric(),
    ArrowType = character(),
    stringsAsFactors = FALSE
  )
  
  # 如果未指定阈值，则自动计算阈值
  if(is.null(threshold_diff)) {
    # 使用数据的分位数来计算阈值
    pred_values <- pred_data[[paste0(model_name, "_pred")]]
    threshold_diff <- quantile(abs(pred_values), 0.25, na.rm = TRUE)
    cat(paste0("自动计算的", model_name, "模型阈值为: ", round(threshold_diff, 4), "\n"))
  }
  
  # 对于每一行数据，如果有多个特征值为1，则表示组合
  for (i in 1:nrow(pred_data)) {
    row <- pred_data[i, ]
    
    # 提取当前特征组合 (值为 1 的特征)
    features_in_combo <- feature_cols[row[feature_cols] == 1]
    pred_value <- row[[paste0(model_name, "_pred")]]
    current_combo_name <- paste(features_in_combo, collapse = ",")
    
    # 只处理有多个特征的组合
    if (length(features_in_combo) > 1) {
      # 遍历当前组合的所有可能的子集 (至少包含一个特征)
      for (j in 1:(length(features_in_combo) - 1)) {
        feature_subsets <- combn(features_in_combo, j, simplify = FALSE)
        
        for (subset in feature_subsets) {
          subset_name <- paste(subset, collapse = ",")
          
          # 查找只包含该子集中特征的行 (子集中的特征为 1，其余特征为 -1)
          subset_pattern <- rep(-1, length(feature_cols))
          names(subset_pattern) <- feature_cols
          subset_pattern[subset] <- 1
          
          subset_rows_indices <- which(apply(pred_data[, feature_cols], 1, function(r) all(r == subset_pattern)))
          
          if (length(subset_rows_indices) == 1) {
            subset_value <- pred_data[subset_rows_indices, paste0(model_name, "_pred")]
            diff_value <- pred_value - subset_value
            
            # 确定互作类型
            interaction <- ifelse(diff_value >= threshold_diff, "Enhancement",
                                  ifelse(diff_value <= -threshold_diff, "Inhibition", "Neutral"))
            
            # 添加到结果中
            results <- rbind(results, data.frame(
              Source = subset_name,
              Target = current_combo_name,
              Difference = diff_value,
              Interaction = interaction,
              Metric_Diff = diff_value,
              ArrowType = ifelse(diff_value > 0, "Enhancement", "Inhibition"),
              stringsAsFactors = FALSE
            ))
          }
        }
      }
    }
  }
  
  # 转换交互类型为有序因子
  results$Interaction <- factor(results$Interaction, levels = c("Inhibition", "Neutral", "Enhancement"))
  results$ArrowType <- factor(results$ArrowType, levels = c("Enhancement", "Inhibition"))
  
  return(results)
}

# 为每个模型生成互作网络
generate_model_interaction_networks <- function(combined_results, feature_cols, threshold_diff = NULL) {
  # 按排序结果选择前100行数据用于网络可视化
  xgb_top <- xgb_sorted[1:min(100, nrow(xgb_sorted)), ]
  lasso_top <- lasso_sorted[1:min(100, nrow(lasso_sorted)), ]
  mlp_top <- mlp_sorted[1:min(100, nrow(mlp_sorted)), ]
  
  # 准备每个模型的互作边数据 - 使用自动阈值
  xgb_edges <- prepare_feature_interaction_data(xgb_top, "xgb", feature_cols, threshold_diff)
  lasso_edges <- prepare_feature_interaction_data(lasso_top, "lasso", feature_cols, threshold_diff)
  mlp_edges <- prepare_feature_interaction_data(mlp_top, "mlp", feature_cols, threshold_diff)
  
  # 绘制所有的模型网络图
  models <- list(xgb = xgb_edges, lasso = lasso_edges, mlp = mlp_edges)
  plots <- list()
  
  for (model_name in names(models)) {
    edges <- models[[model_name]]
    
    # 如果没有显著互作，则跳过
    if (nrow(edges) == 0) {
      message(model_name, " 无显著互作，跳过")
      next
    }
    
    # 自动计算阈值（如果未指定）
    if (is.null(threshold_diff)) {
      auto_threshold <- quantile(abs(edges$Difference), 0.75, na.rm = TRUE)
      cat(paste0("自动计算的", model_name, "模型交互网络阈值为: ", round(auto_threshold, 4), "\n"))
    } else {
      auto_threshold <- threshold_diff
    }
    
    # 过滤显著互作
    sig_edges <- edges %>% filter(abs(Difference) > auto_threshold)
    
    if (nrow(sig_edges) == 0) {
      message(model_name, " 过滤后无显著互作，跳过")
      next
    }
    
    plot_title <- paste("Interaction Network -", toupper(model_name), "Model Predictions")
    
    g <- graph_from_data_frame(sig_edges, directed = TRUE)
    
    # 动态颜色梯度设置
    min_diff <- min(sig_edges$Metric_Diff, na.rm = TRUE)
    max_diff <- max(sig_edges$Metric_Diff, na.rm = TRUE)
    
    if (min_diff < 0 && max_diff > 0) {
      colors <- c("darkred", "white", "darkgreen")
      values <- scales::rescale(c(min_diff, 0, max_diff))
    } else if (min_diff >= 0) {
      colors <- c("white", "darkgreen")
      values <- scales::rescale(c(0, max_diff))
    } else if (max_diff <= 0) {
      colors <- c("darkred", "white")
      values <- scales::rescale(c(min_diff, 0))
    }
    
    # 隐藏网格线和轴，增大字体
    visual_theme <- theme_void() + 
      theme(
        plot.title = element_text(size = 24, face = "bold", hjust = 0.5, margin = margin(b = 20)),
        plot.subtitle = element_text(size = 18, hjust = 0.5, margin = margin(b = 15)),
        legend.position = "right",
        legend.title = element_text(size = 14, face = "bold"),
        legend.text = element_text(size = 12),
        plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
      )
    
    arrow_types_present <- unique(sig_edges$ArrowType)
    linetype_values <- c("Enhancement" = "solid", "Inhibition" = "dashed")
    
    # 只保留数据中存在的类别
    linetype_values <- linetype_values[names(linetype_values) %in% arrow_types_present]
    
    single_plot <- ggraph(g, layout = "fr") +
      geom_edge_link(
        aes(
          edge_alpha = scales::rescale(abs(Metric_Diff), to = c(0.3, 1)),
          edge_width = scales::rescale(abs(Metric_Diff), to = c(1, 4)),
          color = Metric_Diff,
          linetype = ArrowType
        ),
        arrow = arrow(length = unit(0.18, "inches")),
        end_cap = circle(4, "mm"),
        show.legend = TRUE
      ) +
      geom_node_point(
        size = 5,
        shape = 21,
        fill = "#1f78b4",
        color = "black",
        stroke = 1.5
      ) +
      geom_node_text(
        aes(label = name),
        repel = TRUE,
        size = 4,  # 增大标签字体
        color = "#333333",
        fontface = "bold"
      ) +
      scale_edge_color_gradientn(
        colors = colors,
        values = values,
        name = "交互强度\n(负值: 抑制, 正值: 增强)",
        guide = guide_colorbar(barwidth = 1.5, barheight = 10)
      ) +
      scale_edge_width_continuous(range = c(1, 4), guide = "none") +
      scale_edge_alpha_continuous(guide = "none") +
      scale_edge_linetype_manual(
        name = "箭头类型",
        values = linetype_values
      ) +
      labs(
        title = plot_title,
        subtitle = paste0("阈值: ", round(auto_threshold, 4)),
        caption = paste("分析日期:", Sys.Date())
      ) +
      visual_theme
    
    # 保存图形
    ggsave(file.path(net_dir, paste0(model_name, "_interaction_network.pdf")),
           single_plot, width = 16, height = 12, dpi = 600)
    ggsave(file.path(net_dir, paste0(model_name, "_interaction_network.png")),
           single_plot, width = 16, height = 12, dpi = 600)
    
    plots[[model_name]] <- single_plot
  }
  
  # 如果有多于一个图形，则创建组合图
  if (length(plots) > 1) {
    combined_plot <- wrap_plots(plots, ncol = 2) +
      plot_annotation(title = "Model Predictions Interaction Networks",
                      tag_levels = "A",
                      theme = theme(plot.title = element_text(size = 36, face = "bold", hjust = 0.5)))
    
    ggsave(file.path(net_dir, "combined_model_networks.pdf"),
           combined_plot, width = 32, height = 24, dpi = 600)
    ggsave(file.path(net_dir, "combined_model_networks.png"),
           combined_plot, width = 32, height = 24, dpi = 600)
  }
  
  return(plots)
}

# 生成共识互作网络 - 展示多个模型都预测的互作关系
generate_consensus_interaction_network <- function(combined_results, feature_cols, threshold_diff = NULL) {
  # 找出所有模型都预测有提升的组合
  consensus_rows <- which(
    combined_results$lasso_improvement > 0 & 
      combined_results$mlp_improvement > 0 & 
      combined_results$xgb_improvement > 0
  )
  
  # 如果没有共识组合，则返回
  if (length(consensus_rows) == 0) {
    message("没有找到所有模型都预测有提升的组合")
    return(NULL)
  }
  
  # 选择前50个共识组合
  consensus_data <- combined_results[consensus_rows, ]
  consensus_data <- consensus_data[order(-rowMeans(consensus_data[, c("lasso_pred", "mlp_pred", "xgb_pred")])), ]
  consensus_top <- consensus_data[1:min(50, nrow(consensus_data)), ]
  
  # 准备平均预测值的互作边数据
  consensus_top$avg_pred <- rowMeans(consensus_top[, c("lasso_pred", "mlp_pred", "xgb_pred")])
  consensus_edges <- prepare_feature_interaction_data(consensus_top, "avg", feature_cols, threshold_diff)
  
  # 自动计算显著阈值（如果未指定）
  if (is.null(threshold_diff)) {
    auto_threshold <- quantile(abs(consensus_edges$Difference), 0.75, na.rm = TRUE)
    cat(paste0("自动计算的共识模型交互网络阈值为: ", round(auto_threshold, 4), "\n"))
  } else {
    auto_threshold <- threshold_diff
  }
  
  # 如果没有显著互作，则返回
  if (nrow(consensus_edges) == 0 || 
      nrow(consensus_edges %>% filter(abs(Difference) > auto_threshold)) == 0) {
    message("共识数据中无显著互作")
    return(NULL)
  }
  
  # 过滤显著互作
  sig_consensus_edges <- consensus_edges %>% filter(abs(Difference) > auto_threshold)
  
  # 绘制共识网络图
  g <- graph_from_data_frame(sig_consensus_edges, directed = TRUE)
  
  # 动态颜色梯度设置
  min_diff <- min(sig_consensus_edges$Metric_Diff, na.rm = TRUE)
  max_diff <- max(sig_consensus_edges$Metric_Diff, na.rm = TRUE)
  
  if (min_diff < 0 && max_diff > 0) {
    colors <- c("darkred", "white", "darkgreen")
    values <- scales::rescale(c(min_diff, 0, max_diff))
  } else if (min_diff >= 0) {
    colors <- c("white", "darkgreen")
    values <- scales::rescale(c(0, max_diff))
  } else if (max_diff <= 0) {
    colors <- c("darkred", "white")
    values <- scales::rescale(c(min_diff, 0))
  }
  
  # 隐藏网格线和轴，增大字体
  visual_theme <- theme_void() + 
    theme(
      plot.title = element_text(size = 24, face = "bold", hjust = 0.5, margin = margin(b = 20)),
      plot.subtitle = element_text(size = 18, hjust = 0.5, margin = margin(b = 15)),
      legend.position = "right",
      legend.title = element_text(size = 14, face = "bold"),
      legend.text = element_text(size = 12),
      plot.margin = margin(t = 20, r = 20, b = 20, l = 20)
    )
  
  # 修复linetype问题 - 确保数据中存在相应的类别
  arrow_types_present <- unique(sig_consensus_edges$ArrowType)
  linetype_values <- c("Enhancement" = "solid", "Inhibition" = "dashed")
  
  # 只保留数据中存在的类别
  linetype_values <- linetype_values[names(linetype_values) %in% arrow_types_present]
  
  consensus_plot <- ggraph(g, layout = "fr") +
    geom_edge_link(
      aes(
        edge_alpha = scales::rescale(abs(Metric_Diff), to = c(0.3, 1)),
        edge_width = scales::rescale(abs(Metric_Diff), to = c(1, 4)),
        color = Metric_Diff,
        linetype = ArrowType
      ),
      arrow = arrow(length = unit(0.18, "inches")),
      end_cap = circle(4, "mm"),
      show.legend = TRUE
    ) +
    geom_node_point(
      size = 5,
      shape = 21,
      fill = "#1f78b4",
      color = "black",
      stroke = 1.5
    ) +
    geom_node_text(
      aes(label = name),
      repel = TRUE,
      size = 4,  # 增大标签字体
      color = "#333333",
      fontface = "bold"
    ) +
    scale_edge_color_gradientn(
      colors = colors,
      values = values,
      name = "交互强度\n(负值: 抑制, 正值: 增强)",
      guide = guide_colorbar(barwidth = 1.5, barheight = 10)
    ) +
    scale_edge_width_continuous(range = c(1, 4), guide = "none") +
    scale_edge_alpha_continuous(guide = "none") +
    scale_edge_linetype_manual(
      name = "箭头类型",
      values = linetype_values
    ) +
    labs(
      title = "共识互作网络 (所有模型)",
      subtitle = paste0("阈值: ", round(auto_threshold, 4)),
      caption = paste("分析日期:", Sys.Date())
    ) +
    visual_theme
  
  # 保存图形
  ggsave(file.path(net_dir, "consensus_interaction_network.pdf"), 
         consensus_plot, width = 18, height = 14, dpi = 600)
  ggsave(file.path(net_dir, "consensus_interaction_network.png"), 
         consensus_plot, width = 18, height = 14, dpi = 600)
  
  return(consensus_plot)
}

# 互作摘要表 - 针对正向互作
summarize_positive_interactions <- function(combined_results) {
  # 选择所有模型都预测有提升的组合
  consensus_rows <- which(
    combined_results$lasso_improvement > 0 & 
      combined_results$mlp_improvement > 0 & 
      combined_results$xgb_improvement > 0
  )
  
  if (length(consensus_rows) == 0) {
    return(NULL)
  }
  
  consensus_data <- combined_results[consensus_rows, ]
  
  # 计算平均预测提升度
  consensus_data$avg_improvement <- rowMeans(
    consensus_data[, c("lasso_improvement_pct", "mlp_improvement_pct", "xgb_improvement_pct")]
  )
  
  # 按平均提升度排序
  consensus_sorted <- consensus_data[order(-consensus_data$avg_improvement), ]
  
  # 摘要表
  summary_table <- consensus_sorted[, c("feature_combo", 
                                        "lasso_improvement_pct", 
                                        "mlp_improvement_pct", 
                                        "xgb_improvement_pct", 
                                        "avg_improvement")]
  
  # 重命名列以便阅读
  colnames(summary_table) <- c("特征组合", "Lasso提升(%)", "MLP提升(%)", "XGBoost提升(%)", "平均提升(%)")
  
  # 保存摘要表
  write.csv(summary_table, file.path(net_dir, "positive_interactions_summary.csv"))
  
  return(summary_table)
}
