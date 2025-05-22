# ======== 数据读取 ========

# 使用文件选择窗口读取数据集
data_path <- file.choose()  # 弹出文件选择窗口
data_file_name <- basename(data_path)
data_file_name <- gsub(".csv$", "", data_file_name)
raw_data <- read.csv(data_path)  # 读取选择的CSV文件
head(raw_data)  # 显示数据的前几行

# 读完数据再设置环境变量

# ======== 目录创建 ========

# 创建所有必要的目录并获取完整路径
paths <- setup_directories(config)

# ======== 预处理 ========

# 调用函数处理data
raw_data_result <- process_data(raw_data, 
                                binary_cols = 1:(ncol(raw_data)-1), 
                                target_cols = ncol(raw_data):ncol(raw_data),
                                replace_zeros_with = -1)
head(raw_data_result)

# 查看处理结果
head(raw_data_result$binary_matrix)
head(raw_data_result$target_values)

# -------- 生成交互特征 --------

raw_data_result$interaction_features <- 
  generate_interaction_features(raw_data_result$binary_matrix, 
                                                      max_order = config$feature_engineering$max_interaction_order,
                                                      cache_path = paths$data_dir,
                                                      cache_id = data_file_name)

# 交互特征列表
raw_data_result$interaction_species_names <- colnames(raw_data_result$interaction_features)
raw_data_result$max_order <- config$feature_engineering$max_interaction_order
head(raw_data_result$interaction_features)
raw_data_result$interaction_species_names

# -------- 数据集划分 --------

modelset_data <- split_dataset(raw_data_result$interaction_features,
                            raw_data_result$target_values,
                            train_ratio = 0.8,
                            val_ratio = 0,
                            test_ratio = 0.2)

X_train <- modelset_data$X_train
y_train <- as.numeric(modelset_data$y_train)  # 确保是向量
X_test <- modelset_data$X_test
y_test <- as.numeric(modelset_data$y_test)    # 确保是向量


print(paste("训练集大小:", modelset_data$sizes$train))
print(paste("测试集大小:", modelset_data$sizes$test))



# ========= 特征选择 ========

#print("执行特征选择...")

# 使用三种不同方法进行特征选择
#top_n <- 50  # 选择前50个特征

# Lasso特征选择
#print("使用Lasso方法进行特征选择...")
#lasso_features <- select_features(X_train, y_train, method = "lasso", top_n = top_n)

# 相关性特征选择
#print("使用相关性方法进行特征选择...")
#cor_features <- select_features(X_train, y_train, method = "correlation", top_n = top_n)

# 方差特征选择
#print("使用方差方法进行特征选择...")
#var_features <- select_features(X_train, y_train, method = "variance", top_n = top_n)

# 检查三种方法选择的特征有多少重叠
#common_features <- intersect(intersect(lasso_features, cor_features), var_features)
#print(paste("三种方法共同选择的特征数量:", length(common_features)))

# 为后续建模选择特征子集 (这里我们使用Lasso选择的特征)
#selected_features <- lasso_features
#X_train_selected <- X_train[, selected_features, drop = FALSE]
#X_test_selected <- X_test[, selected_features, drop = FALSE]

#print(paste("选择的特征数量:", length(selected_features)))

# ======== 模型训练 ========

# -------- Lasso回归模型 ---------
print("\n训练Lasso模型...")
lasso_cv_results <- train_with_cv(
  X = X_train,
  y = y_train,
  model_name = "lasso",
  params = config$model_params$lasso,
  k = config$cv_params$k_folds,
  random_seed = 123
)

# 从交叉验证中选择最佳Lasso模型
best_lasso_idx <- which.min(sapply(lasso_cv_results$metrics$by_fold, 
                                   function(x) x$val$rmse))
best_lasso_model <- lasso_cv_results$models[[best_lasso_idx]]

print(paste("最佳Lasso模型 (折", best_lasso_idx, ")"))
print(paste("  验证RMSE:", round(lasso_cv_results$metrics$by_fold[[best_lasso_idx]]$val$rmse, 4)))
print(paste("  验证R²:", round(lasso_cv_results$metrics$by_fold[[best_lasso_idx]]$val$r2, 4)))

saveRDS(lasso_cv_results,
        file = file.path(paths$model_dir,"lasso_cv_results.rds"))
saveRDS(best_lasso_model,
        file = file.path(paths$model_dir,"best_lasso_model.rds"))

# -------- MLP神经网络模型 --------
print("\n训练MLP模型...")
mlp_cv_results <- train_with_cv(
  X = X_train,
  y = y_train,
  model_name = "mlp",
  params = config$model_params$mlp,
  k = config$cv_params$k_folds,
  random_seed = 123
)

# 从交叉验证中选择最佳MLP模型
best_mlp_idx <- which.min(sapply(mlp_cv_results$metrics$by_fold, 
                                 function(x) x$val$rmse))
best_mlp_model <- mlp_cv_results$models[[best_mlp_idx]]

print(paste("最佳MLP模型 (折", best_mlp_idx, ")"))
print(paste("  验证RMSE:", round(mlp_cv_results$metrics$by_fold[[best_mlp_idx]]$val$rmse, 4)))
print(paste("  验证R²:", round(mlp_cv_results$metrics$by_fold[[best_mlp_idx]]$val$r2, 4)))

saveRDS(mlp_cv_results,
        file = file.path(paths$model_dir,"mlp_cv_results.rds"))
saveRDS(best_mlp_model,
        file = file.path(paths$model_dir,"best_mlp_model.rds"))

# -------- XGBoost模型 --------
print("\n训练XGBoost模型...")

xgb_cv_results <- train_with_cv(
  X = X_train,
  y = y_train,
  model_name = "xgboost",
  params = config$model_params$xgboost,
  k = config$cv_params$k_folds,
  random_seed = 123
)

# 从交叉验证中选择最佳XGBoost模型
best_xgb_idx <- which.min(sapply(xgb_cv_results$metrics$by_fold, 
                                 function(x) x$val$rmse))
best_xgb_model <- xgb_cv_results$models[[best_xgb_idx]]

print(paste("最佳XGBoost模型 (折", best_xgb_idx, ")"))
print(paste("  验证RMSE:", round(xgb_cv_results$metrics$by_fold[[best_xgb_idx]]$val$rmse, 4)))
print(paste("  验证R²:", round(xgb_cv_results$metrics$by_fold[[best_xgb_idx]]$val$r2, 4)))

saveRDS(xgb_cv_results,
        file = file.path(paths$model_dir,"xgb_cv_results.rds"))
saveRDS(best_xgb_model,
        file = file.path(paths$model_dir,"best_xgb_model.rds"))

# ======== 在测试集上评估模型 ========
print("\n在测试集上评估模型性能...")

# -------- 评估Lasso模型 --------
X_test_scaled_lasso <- scale(X_test, 
                             center = best_lasso_model$scaler$means, 
                             scale = best_lasso_model$scaler$sds)
lasso_test_pred <- predict(best_lasso_model$model, newx = X_test_scaled_lasso)
lasso_test_metrics <- calculate_metrics(y_test, lasso_test_pred, config$evaluation$test_eval_metrics)

# -------- 评估MLP模型 --------
X_test_scaled_mlp <- scale(X_test, 
                           center = best_mlp_model$scaler$means, 
                           scale = best_mlp_model$scaler$sds)
mlp_test_pred <- best_mlp_model$model %>% predict(X_test_scaled_mlp)
mlp_test_metrics <- calculate_metrics(y_test, mlp_test_pred, config$evaluation$test_eval_metrics)

# -------- 评估XGBoost模型 --------
dtest <- xgb.DMatrix(data = as.matrix(X_test), label = y_test)
xgb_test_pred <- predict(best_xgb_model$model, dtest)
xgb_test_metrics <- calculate_metrics(y_test, xgb_test_pred, config$evaluation$test_eval_metrics)

# ======== 比较模型性能 ========

print("\n模型性能比较 (测试集):")

model_comparison <- data.frame(
  data = data_file_name,
  Model = c("Lasso", "MLP", "XGBoost"),
  RMSE = c(lasso_test_metrics$rmse, mlp_test_metrics$rmse, xgb_test_metrics$rmse),
  MAE = c(lasso_test_metrics$mae, mlp_test_metrics$mae, xgb_test_metrics$mae),
  R2 = c(lasso_test_metrics$r2, mlp_test_metrics$r2, xgb_test_metrics$r2),
  MAPE = c(lasso_test_metrics$mape, mlp_test_metrics$mape, xgb_test_metrics$mape)
)

print(model_comparison)

write.csv(model_comparison, 
       file.path(paths$data_dir, "model_comparison.csv"))

# 找出性能最好的模型
best_model_idx <- which.min(model_comparison$RMSE)
best_model_name <- model_comparison$Model[best_model_idx]
print(paste("\n在测试集上性能最好的模型是:", best_model_name))
print(paste("RMSE:", round(model_comparison$RMSE[best_model_idx], 4)))
print(paste("R²:", round(model_comparison$R2[best_model_idx], 4)))

# ======== 特征重要性分析 ========

print("\n分析特征重要性...")

# -------- 针对Lasso模型查看特征重要性 --------
#if (best_model_name == "Lasso") {
  lasso_importance <- best_lasso_model$feature_importance
  top_lasso_features <- head(lasso_importance, 10)
  print("Lasso模型最重要的10个特征:")
  print(top_lasso_features)
#}
write.csv(lasso_importance,
          file.path(paths$data_dir,paste0(data_file_name,"_lasso_importance.csv")))

lasso_imp_plot <- plot_feature_importance(lasso_importance,
                              top_n = 15,
                              title = paste0(data_file_name,"Features Importance by Lasso"),
                              filename = paste0(data_file_name,"_Features_Importance_Lasso"))
  
# -------- 针对MLP模型查看特征重要性 --------
# MLP模型的特征重要性分析
print("\n分析MLP模型的特征重要性...")

# 排列重要性方法 - 适用于Keras模型
calculate_permutation_importance <- function(model_obj, 
                                             X, 
                                             y, 
                                             metric = "rmse", 
                                             n_repeats = 3) {
  # 从model_obj中提取实际的keras模型
  keras_model <- model_obj$model
  
  # 原始预测函数
  predict_fn <- function(X_data) {
    # 确保数据类型正确
    if (is.data.frame(X_data)) {
      X_matrix <- as.matrix(X_data)
    } else {
      X_matrix <- X_data
    }
    
    # 使用keras模型预测
    preds <- as.vector(keras_model %>% predict(X_matrix))
    return(preds)
  }
  
  # 原始性能
  original_pred <- predict_fn(X)
  if (metric == "rmse") {
    original_perf <- sqrt(mean((original_pred - y)^2))
  } else if (metric == "mae") {
    original_perf <- mean(abs(original_pred - y))
  }
  
  # 存储每个特征的重要性
  n_features <- ncol(X)
  importance <- numeric(n_features)
  importance_std <- numeric(n_features)
  
  # 对每个特征进行排列
  for (i in 1:n_features) {
    feature_importances <- numeric(n_repeats)
    
    for (r in 1:n_repeats) {
      # 复制数据
      X_permuted <- X
      # 打乱特定列
      X_permuted[,i] <- sample(X_permuted[,i])
      
      # 使用打乱后的数据预测
      permuted_pred <- predict_fn(X_permuted)
      if (metric == "rmse") {
        permuted_perf <- sqrt(mean((permuted_pred - y)^2))
      } else if (metric == "mae") {
        permuted_perf <- mean(abs(permuted_pred - y))
      }
      
      # 重要性是性能下降的程度
      feature_importances[r] <- permuted_perf - original_perf
    }
    
    importance[i] <- mean(feature_importances)
    importance_std[i] <- sd(feature_importances)
  }
  
  # 创建数据框，包含特征名称和重要性分数
  feature_importance <- data.frame(
    feature = colnames(X),
    importance = importance,
    std_dev = importance_std
  )
  
  # 按重要性降序排列
  feature_importance <- feature_importance[order(-feature_importance$importance),]
  return(feature_importance)
}

# 使用上面定义的函数计算MLP模型的特征重要性
print("计算MLP特征重要性，这可能需要一些时间...")
mlp_importance <- calculate_permutation_importance(best_mlp_model, X_test, y_test, n_repeats = 5)
top_mlp_features <- head(mlp_importance, 10)
print("MLP模型最重要的10个特征:")
print(top_mlp_features)

write.csv(mlp_importance,
          file.path(paths$data_dir,paste0(data_file_name,"_mlp_importance.csv")))

# 可视化特征重要性
#if (requireNamespace("ggplot2", quietly = TRUE)) {
#  library(ggplot2)
#  
#  # 选择前10个最重要的特征
#  top_n <- 10
#  plot_data <- head(mlp_importance, top_n)
#  
#  # 重新排序特征以便按照重要性从高到低显示
#  plot_data$feature <- factor(plot_data$feature, 
#                             levels = plot_data$feature[order(plot_data$importance)])
#  
#  # 创建条形图
#  p <- ggplot(plot_data, aes(x = importance, y = feature)) +
#    geom_bar(stat = "identity", fill = "steelblue", alpha = 0.8) +
#    geom_errorbar(aes(xmin = importance - std_dev, xmax = importance + std_dev), 
#                 width = 0.2, color = "darkred") +
#    labs(title = "MLP模型特征重要性",
#         subtitle = "使用排列重要性方法",
#         x = "重要性分数（误差增加量）",
#         y = "特征") +
#    theme_minimal() +
#    theme(axis.text.y = element_text(size = 10),
#          plot.title = element_text(size = 14, face = "bold"))
#  
#  print(p)
#}

mlp_imp_plot <- plot_feature_importance(mlp_importance,
                                        top_n = 15,
                                        title = paste0(data_file_name,"Features Importance by MLP"),
                                        filename = paste0(data_file_name,"_Features_Importance_MLP"))


# -------- 针对XGBoost模型查看特征重要性 --------
#if (best_model_name == "XGBoost") {
  xgb_importance <- best_xgb_model$feature_importance
  top_xgb_features <- head(xgb_importance, 10)
  print("XGBoost模型最重要的10个特征:")
  print(top_xgb_features)
#}

write.csv(xgb_importance,
          file.path(paths$data_dir,paste0(data_file_name,"_xgb_importance.csv")))
xgb_Gain <- xgb_importance[,2:2]
colnames(xgb_importance)[colnames(xgb_importance) == "Gain" ] <- "importance"
colnames(xgb_importance)[colnames(xgb_importance) == "Feature" ] <- "feature"
xgb_importance <- cbind(xgb_importance,xgb_Gain)
xgb_imp_plot <- plot_feature_importance(xgb_importance,
                                        top_n = 15,
                                        title = paste0(data_file_name,"Features Importance by XGBoost"),
                                        filename = paste0(data_file_name,"_Features_Importance_XGBoost"))
  

# ======== 可视化预测结果 ========

print("\n生成预测vs实际值散点图...")

# 创建预测结果数据框
predictions_df <- data.frame(
  Actual = y_test,
  Lasso = as.numeric(lasso_test_pred),
  MLP = as.numeric(mlp_test_pred),
  XGBoost = as.numeric(xgb_test_pred)
)

# -------- Lasso --------
lasso_predictions_df <- data.frame(
  actual = y_test,
  predicted = as.numeric(lasso_test_pred))
lasso_test_pred_plot <- plot_actual_vs_predicted(lasso_predictions_df,
                                                 title = paste0(data_file_name," Actual vs. Predicted Values by lasso"),
                                                 filename =paste0(data_file_name,"_Actual_vs_Predicted_Values_lasso"))
lasso_residuals_plot <- plot_residuals(lasso_predictions_df,
                                       title = paste0(data_file_name," Residual Plot by lasso"),
                                       filename =paste0(data_file_name,"_Residual_Plot_lasso"))

# -------- MLP --------
mlp_predictions_df <- data.frame(
  actual = y_test,
  predicted = as.numeric(mlp_test_pred))
mlp_test_pred_plot <- plot_actual_vs_predicted(mlp_predictions_df,
                                                 title = paste0(data_file_name," Actual vs. Predicted Values by MLP"),
                                                 filename =paste0(data_file_name,"_Actual_vs_Predicted_Values_MLP"))
mlp_residuals_plot <- plot_residuals(mlp_predictions_df,
                                       title = paste0(data_file_name," Residual Plot by MLP"),
                                       filename =paste0(data_file_name,"_Residual_Plot_MLP"))
mlp_history_plot <- plot_training_history(best_mlp_model$history_df,
                                          title = paste0(data_file_name," MLP training history"),
                                          filename =paste0(data_file_name,"_training_history_MLP"))

# -------- XGBoost --------
xgb_predictions_df <- data.frame(
  actual = y_test,
  predicted = as.numeric(xgb_test_pred))
xgb_test_pred_plot <- plot_actual_vs_predicted(xgb_predictions_df,
                                                 title = paste0(data_file_name," Actual vs. Predicted Values by XGBoost"),
                                                 filename =paste0(data_file_name,"_Actual_vs_Predicted_Values_XGBoost"))
xgb_residuals_plot <- plot_residuals(xgb_predictions_df,
                                     title = paste0(data_file_name," Residual Plot by XGBoost"),
                                     filename =paste0(data_file_name,"_Residual_Plot_XGBoost"))

colnames(best_xgb_model[["train_log"]])[1] <- "epoch"
xgb_history_plot <- plot_training_history(best_xgb_model$train_log,
                                          title = paste0(data_file_name," XGBoost training history"),
                                          x_label = "iter",
                                          y_label = "RMSE",
                                          metric_labels = c("Training RMSE", "Validation RMSE"),
                                          filename =paste0(data_file_name,"_training_history_XGBoost"))

print("建模和评估流程完成！")

# -------- 创建特征比较 --------
# 合并所有模型的前15个特征
tryCatch({
  # 确保所有模型的特征重要性数据都准备好了
  if (exists("lasso_importance") && exists("xgb_importance") && exists("mlp_importance")) {
    # 创建比较数据框
    compare_features <- function(lasso_imp, xgb_imp, mlp_imp, top_n = 15) {
      # 提取每个模型的前top_n个特征
      lasso_top <- head(lasso_imp[order(-lasso_imp$importance), ], top_n)
      xgb_top <- head(xgb_imp[order(-xgb_imp$importance), ], top_n)
      mlp_top <- head(mlp_imp[order(-mlp_imp$importance), ], top_n)
      
      # 合并所有独特的特征
      all_features <- unique(c(lasso_top$feature, xgb_top$feature, mlp_top$feature))
      
      # 创建比较数据框
      comparison <- data.frame(feature = all_features)
      
      # 添加每个模型的重要性
      comparison$lasso_importance <- sapply(all_features, function(f) {
        idx <- which(lasso_imp$feature == f)
        if (length(idx) > 0) lasso_imp$importance[idx[1]] else 0
      })
      
      comparison$xgb_importance <- sapply(all_features, function(f) {
        idx <- which(xgb_imp$feature == f)
        if (length(idx) > 0) xgb_imp$importance[idx[1]] else 0
      })
      
      comparison$mlp_importance <- sapply(all_features, function(f) {
        idx <- which(mlp_imp$feature == f)
        if (length(idx) > 0) mlp_imp$importance[idx[1]] else 0
      })
      
      # 归一化重要性分数
      comparison$lasso_importance <- comparison$lasso_importance / max(comparison$lasso_importance)
      comparison$xgb_importance <- comparison$xgb_importance / max(comparison$xgb_importance)
      comparison$mlp_importance <- comparison$mlp_importance / max(comparison$mlp_importance)
      
      # 按平均重要性排序
      comparison$avg_importance <- (comparison$lasso_importance + 
                                      comparison$xgb_importance + 
                                      comparison$mlp_importance) / 3
      
      comparison <- comparison[order(-comparison$avg_importance), ]
      return(comparison)
    }
    
    # 创建特征比较
    feature_comparison <- compare_features(lasso_importance, xgb_importance, mlp_importance)
    print("不同模型特征重要性比较:")
    print(head(feature_comparison, 15))
    
    # 绘制比较图表
    top_compare <- head(feature_comparison, 15)
    comparison_long <- reshape2::melt(top_compare, id.vars = "feature", 
                                      measure.vars = c("lasso_importance", "xgb_importance", "mlp_importance"),
                                      variable.name = "model", value.name = "importance")
    
    # 将模型名称变得更易读
    comparison_long$model <- gsub("_importance", "", comparison_long$model)
    comparison_long$model <- toupper(comparison_long$model)
    
    # 重新排序特征
    comparison_long$feature <- factor(comparison_long$feature, 
                                      levels = top_compare$feature)
    
    # 创建比较条形图
    p_compare <- ggplot(comparison_long, aes(x = importance, y = feature, fill = model)) +
      geom_bar(stat = "identity", position = position_dodge(), alpha = 0.8) +
      scale_fill_manual(values = c("LASSO" = "coral3", "XGB" = "forestgreen", "MLP" = "steelblue")) +
      labs(title = paste0(data_file_name,"数据","模型特征重要性比较"),
           subtitle = "归一化后的重要性分数",
           x = "归一化重要性分数",
           y = "特征",
           fill = "模型") +
      theme_minimal() +
      theme(axis.text.y = element_text(size = 10),
            plot.title = element_text(size = 14, face = "bold"))
    
    print(p_compare)
    
    # 保存比较图表
    ggsave(file.path(paths$plot_dir, paste0(data_file_name,"_feature_importance_comparison.png")), 
           plot = p_compare, width = 12, height = 8, dpi = 300)
    cat("特征重要性比较图已保存\n")
  } else {
    cat("无法创建特征比较，因为一个或多个模型的特征重要性数据不可用\n")
  }
}, error = function(e) {
  cat("创建特征比较时出错:", e$message, "\n")
})

# ======== 特征组合预测 ========

# 生成最多含n个特征的所有组合
new_data <- generate_feature_combinations(raw_data_result$feature_names,
                                          max_length = NULL)

# 打印结果的前10行
head(new_data, 10)

# 打印组合总数
cat("组合总数:", nrow(new_data), "\n")

if (nrow(new_data) <= 10000) {
  write.csv(new_data, file.path(paths$feature_cache, paste0(data_file_name, "_new_data.csv")))
} else {
  # 使用数据表格而不是标准数据框
  new_data_dt <- data.table(new_data)
  
  # 直接写入，利用arrow包的并行性
  write_parquet(
    new_data_dt,
    file.path(paths$feature_cache, paste0(data_file_name, "_new_data.parquet")),
    compression = "snappy",  # snappy更快但压缩率低一些
    chunk_size = 500000,     # 控制每个chunk的大小
    version = "2.6"          # 使用最新版本的parquet格式
  )
}

# 手动删除旧缓存文件
cache_file <- file.path(paths$feature_cache, paste0("new_interaction_", data_file_name, "_order2.rds"))
if (file.exists(cache_file)) file.remove(cache_file)

new_interaction_data <- generate_interaction_features(new_data, 
                              max_order = 2,
                              cache_path = paths$feature_cache,
                              cache_id = paste0("new_interaction_",data_file_name))

if (nrow(new_data) <= 10000) {
  write.csv(new_interaction_data, file.path(paths$feature_cache, paste0(data_file_name, "_new_interaction_data.csv")))
} else {
  # 使用数据表格而不是标准数据框
  new_interaction_data_dt <- data.table(new_interaction_data)
  
  # 直接写入，利用arrow包的并行性
  write_parquet(
    new_interaction_data_dt,
    file.path(paths$feature_cache, paste0(data_file_name, "_new_interaction_data.parquet")),
    compression = "snappy",  # snappy更快但压缩率低一些
    chunk_size = 500000,     # 控制每个chunk的大小
    version = "2.6"          # 使用最新版本的parquet格式
  )
}

models <- list(
  lasso = best_lasso_model,
  mlp = best_mlp_model,
  xgboost = best_xgb_model
)

# 确保数据是数据框
new_data <- as.data.frame(new_data)
new_interaction_data <- as.data.frame(new_interaction_data)

# -------- Lasso --------
# 1. 确保列名匹配
colnames(new_interaction_data) <- names(best_lasso_model$scaler$means)

# 2. 标准化
new_interaction_data_scaled <- scale(new_interaction_data, 
                                     center = best_lasso_model$scaler$means, 
                                     scale = best_lasso_model$scaler$sds)

# 3. 执行预测
lasso_new_data_predictions <- predict(best_lasso_model$model, newx = as.matrix(new_interaction_data_scaled))

# 4. 创建结果数据框 - 使用原始数据而不是标准化数据
lasso_new_data_results <- cbind(as.data.frame(new_interaction_data), lasso_pred = as.numeric(lasso_new_data_predictions))

# 5. 保存结果
write.csv(lasso_new_data_results, file.path(paths$feature_cache, "lasso_direct_predictions.csv"))

# 6. 按预测值降序排序 - 修复这里
lasso_new_data_sorted <- lasso_new_data_results[order(lasso_new_data_results$lasso_pred, decreasing = TRUE), ]

# 7. 保存排序结果
write.csv(lasso_new_data_sorted, file.path(paths$feature_cache, "lasso_direct_sorted.csv"))

# 8. 显示排序结果前几行
cat("\nLasso模型预测的前10个最佳组合:\n")
head(lasso_new_data_sorted[, c(raw_data_result$feature_names, "lasso_pred")], 10)

# -------- MLP --------
# 1. 确保列名匹配
colnames(new_interaction_data) <- names(best_mlp_model$scaler$means)

# 2. 标准化
new_interaction_data_scaled <- scale(new_interaction_data, 
                                     center = best_mlp_model$scaler$means, 
                                     scale = best_mlp_model$scaler$sds)

# 3. 执行预测
mlp_new_data_predictions <- predict(best_mlp_model$model, as.matrix(new_interaction_data_scaled))

# 4. 创建结果数据框 - 使用原始数据
mlp_new_data_results <- cbind(as.data.frame(new_interaction_data), mlp_pred = as.numeric(mlp_new_data_predictions))

# 5. 保存结果
write.csv(mlp_new_data_results, file.path(paths$feature_cache, "mlp_direct_predictions.csv"))

# 6. 按预测值降序排序
mlp_new_data_sorted <- mlp_new_data_results[order(mlp_new_data_results$mlp_pred, decreasing = TRUE), ]

# 7. 保存排序结果
write.csv(mlp_new_data_sorted, file.path(paths$feature_cache, "mlp_direct_sorted.csv"))

# 8. 显示排序结果前几行
cat("\nMLP模型预测的前10个最佳组合:\n")
head(mlp_new_data_sorted[, c(raw_data_result$feature_names, "mlp_pred")], 10)

# -------- XGBoost --------
# 1. 创建DMatrix
xgb_new_data_dmatrix <- xgb.DMatrix(data = as.matrix(new_interaction_data))

# 2. 执行预测
xgb_new_data_predictions <- predict(best_xgb_model$model, xgb_new_data_dmatrix)

# 3. 创建结果数据框
xgb_new_data_results <- cbind(as.data.frame(new_interaction_data), xgb_pred = as.numeric(xgb_new_data_predictions))

# 4. 保存结果
write.csv(xgb_new_data_results, file.path(paths$feature_cache, "xgb_direct_predictions.csv"))

# 5. 按预测值降序排序
xgb_new_data_sorted <- xgb_new_data_results[order(xgb_new_data_results$xgb_pred, decreasing = TRUE), ]

# 6. 保存排序结果
write.csv(xgb_new_data_sorted, file.path(paths$feature_cache, "xgb_direct_sorted.csv"))

# 7. 显示排序结果前几行
cat("\nXGBoost模型预测的前10个最佳组合:\n")
head(xgb_new_data_sorted[, c(raw_data_result$feature_names, "xgb_pred")], 10)

# -------- 合并所有模型结果并计算提升度 --------

# 1. 首先找到原始数据中的最大适应度值
original_max_fitness <- max(raw_data_result$target_values)
cat("\n原始数据中的最大适应度值:", original_max_fitness, "\n")

# 2. 创建合并的结果数据框
combined_new_data_results <- as.data.frame(new_interaction_data)
combined_new_data_results$lasso_pred <- lasso_new_data_results$lasso_pred
combined_new_data_results$mlp_pred <- mlp_new_data_results$mlp_pred
combined_new_data_results$xgb_pred <- xgb_new_data_results$xgb_pred

# 3. 计算每个模型预测的提升度
combined_new_data_results$lasso_improvement <- (combined_new_data_results$lasso_pred / original_max_fitness) - 1
combined_new_data_results$mlp_improvement <- (combined_new_data_results$mlp_pred / original_max_fitness) - 1
combined_new_data_results$xgb_improvement <- (combined_new_data_results$xgb_pred / original_max_fitness) - 1

# 4. 将提升度转换为百分比
combined_new_data_results$lasso_improvement_pct <- combined_new_data_results$lasso_improvement * 100
combined_new_data_results$mlp_improvement_pct <- combined_new_data_results$mlp_improvement * 100
combined_new_data_results$xgb_improvement_pct <- combined_new_data_results$xgb_improvement * 100

# 5. 保存合并结果
write.csv(combined_new_data_results, file.path(paths$feature_cache, "combined_predictions_with_improvement.csv"))

# 6. 分别按各个模型的提升度排序
lasso_sorted <- combined_new_data_results[order(combined_new_data_results$lasso_improvement_pct, decreasing = TRUE), ]
mlp_sorted <- combined_new_data_results[order(combined_new_data_results$mlp_improvement_pct, decreasing = TRUE), ]
xgb_sorted <- combined_new_data_results[order(combined_new_data_results$xgb_improvement_pct, decreasing = TRUE), ]

# 7. 保存各个排序结果
write.csv(lasso_sorted, file.path(paths$feature_cache, "lasso_sorted_by_improvement.csv"))
write.csv(mlp_sorted, file.path(paths$feature_cache, "mlp_sorted_by_improvement.csv"))
write.csv(xgb_sorted, file.path(paths$feature_cache, "xgb_sorted_by_improvement.csv"))

# 8. 显示各模型按提升度排序的前10个组合
cat("\nLasso模型 - 按提升度排序的前10个最佳组合:\n")
print(head(lasso_sorted[, c(raw_data_result$feature_names, "lasso_pred", "lasso_improvement_pct")], 10))

cat("\nMLP模型 - 按提升度排序的前10个最佳组合:\n")
print(head(mlp_sorted[, c(raw_data_result$feature_names, "mlp_pred", "mlp_improvement_pct")], 10))

cat("\nXGBoost模型 - 按提升度排序的前10个最佳组合:\n")
print(head(xgb_sorted[, c(raw_data_result$feature_names, "xgb_pred", "xgb_improvement_pct")], 10))

# 9. 计算超过原始最大值的组合数量 (针对每个模型)
lasso_improvements <- sum(combined_new_data_results$lasso_improvement > 0)
mlp_improvements <- sum(combined_new_data_results$mlp_improvement > 0)
xgb_improvements <- sum(combined_new_data_results$xgb_improvement > 0)

cat("\n预测超过原始最大值的组合数量:\n")
cat("- Lasso模型:", lasso_improvements, 
    "(占总数的", round(lasso_improvements/nrow(combined_new_data_results)*100, 2), "%)\n")
cat("- MLP模型:", mlp_improvements, 
    "(占总数的", round(mlp_improvements/nrow(combined_new_data_results)*100, 2), "%)\n")
cat("- XGBoost模型:", xgb_improvements, 
    "(占总数的", round(xgb_improvements/nrow(combined_new_data_results)*100, 2), "%)\n")

# 10. 各个模型预测提升度最大的组合
lasso_max_idx <- which.max(combined_new_data_results$lasso_improvement)
mlp_max_idx <- which.max(combined_new_data_results$mlp_improvement)
xgb_max_idx <- which.max(combined_new_data_results$xgb_improvement)

cat("\n各模型预测提升度最大的组合:\n")
cat("- Lasso模型最大提升度组合:\n")
print(combined_new_data_results[lasso_max_idx, 
                                c(raw_data_result$feature_names, "lasso_pred", "lasso_improvement_pct")])

cat("\n- MLP模型最大提升度组合:\n")
print(combined_new_data_results[mlp_max_idx, 
                                c(raw_data_result$feature_names, "mlp_pred", "mlp_improvement_pct")])

cat("\n- XGBoost模型最大提升度组合:\n")
print(combined_new_data_results[xgb_max_idx, 
                                c(raw_data_result$feature_names, "xgb_pred", "xgb_improvement_pct")])

# 11. 查找所有模型都预测有提升的组合
all_models_improvement <- combined_new_data_results$lasso_improvement > 0 & 
  combined_new_data_results$mlp_improvement > 0 & 
  combined_new_data_results$xgb_improvement > 0

cat("\n所有三个模型都预测有提升的组合数量:", sum(all_models_improvement), "\n")

# 如果存在这样的组合，显示前几个
if (sum(all_models_improvement) > 0) {
  common_improvements <- combined_new_data_results[all_models_improvement, ]
  
  # 基于三个模型提升度的最小值排序 (保守估计)
  common_improvements$min_improvement <- pmin(
    common_improvements$lasso_improvement,
    common_improvements$mlp_improvement,
    common_improvements$xgb_improvement
  )
  
  common_sorted <- common_improvements[order(common_improvements$min_improvement, decreasing = TRUE), ]
  
  cat("\n所有模型都预测有提升的前10个组合 (按最小提升度排序):\n")
  print(head(common_sorted[, c(raw_data_result$feature_names, 
                               "lasso_improvement_pct", "mlp_improvement_pct", 
                               "xgb_improvement_pct")], 10))
  
  # 保存共同提升的组合
  write.csv(common_sorted, file.path(paths$feature_cache, "all_models_improvement.csv"))
}

# 12. 计算模型之间的一致性
# 查找各个模型预测的前10个组合的重合度
lasso_top10_indices <- order(combined_new_data_results$lasso_pred, decreasing = TRUE)[1:10]
mlp_top10_indices <- order(combined_new_data_results$mlp_pred, decreasing = TRUE)[1:10]
xgb_top10_indices <- order(combined_new_data_results$xgb_pred, decreasing = TRUE)[1:10]

# 计算重合度
lasso_mlp_overlap <- length(intersect(lasso_top10_indices, mlp_top10_indices))
lasso_xgb_overlap <- length(intersect(lasso_top10_indices, xgb_top10_indices))
mlp_xgb_overlap <- length(intersect(mlp_top10_indices, xgb_top10_indices))
all_models_overlap <- length(Reduce(intersect, list(lasso_top10_indices, mlp_top10_indices, xgb_top10_indices)))

cat("\n不同模型预测的前10个组合的重合度:\n")
cat("- Lasso与MLP重合:", lasso_mlp_overlap, "个组合\n")
cat("- Lasso与XGBoost重合:", lasso_xgb_overlap, "个组合\n")
cat("- MLP与XGBoost重合:", mlp_xgb_overlap, "个组合\n")
cat("- 三个模型共同预测的组合:", all_models_overlap, "个组合\n")

# 如果存在三模型共同预测的组合，显示这些组合
if (all_models_overlap > 0) {
  common_indices <- Reduce(intersect, list(lasso_top10_indices, mlp_top10_indices, xgb_top10_indices))
  common_combinations <- combined_new_data_results[common_indices, ]
  
  cat("\n所有三个模型共同预测的前", all_models_overlap, "个最佳组合:\n")
  print(common_combinations[, c(raw_data_result$feature_names, 
                                "lasso_pred", "mlp_pred", "xgb_pred",
                                "lasso_improvement_pct", "mlp_improvement_pct", "xgb_improvement_pct")])
}



# -------- 添加特征组合描述列 --------

# 使用raw_data_result中的feature_names
feature_names <- raw_data_result$feature_names

# 为各个结果添加特征组合描述列
xgb_new_data_results$feature_combo <- apply(xgb_new_data_results[, feature_names], 1, create_feature_combination_label, feature_names=raw_data_result$feature_names)
lasso_new_data_results$feature_combo <- apply(lasso_new_data_results[, feature_names], 1, create_feature_combination_label, feature_names=raw_data_result$feature_names)
mlp_new_data_results$feature_combo <- apply(mlp_new_data_results[, feature_names], 1, create_feature_combination_label, feature_names=raw_data_result$feature_names)
combined_new_data_results$feature_combo <- apply(combined_new_data_results[, feature_names], 1, create_feature_combination_label, feature_names=raw_data_result$feature_names)

# 保存更新后的结果文件
write.csv(xgb_new_data_results, file.path(paths$feature_cache, "xgb_direct_predictions_with_labels.csv"))
write.csv(lasso_new_data_results, file.path(paths$feature_cache, "lasso_direct_predictions_with_labels.csv"))
write.csv(mlp_new_data_results, file.path(paths$feature_cache, "mlp_direct_predictions_with_labels.csv"))
write.csv(combined_new_data_results, file.path(paths$feature_cache, "combined_predictions_with_labels.csv"))

# 按预测值降序重新排序（因为添加了新列）
xgb_sorted_labeled <- xgb_new_data_results[order(xgb_new_data_results$xgb_pred, decreasing = TRUE), ]
lasso_sorted_labeled <- lasso_new_data_results[order(lasso_new_data_results$lasso_pred, decreasing = TRUE), ]
mlp_sorted_labeled <- mlp_new_data_results[order(mlp_new_data_results$mlp_pred, decreasing = TRUE), ]

# 显示包含特征组合标签的排序结果
cat("\nLasso模型预测的前10个最佳组合 (带特征标签):\n")
print(head(lasso_sorted_labeled[, c(raw_data_result$feature_names, "lasso_pred", "feature_combo")], 10))

cat("\nMLP模型预测的前10个最佳组合 (带特征标签):\n")
print(head(mlp_sorted_labeled[, c(raw_data_result$feature_names, "mlp_pred", "feature_combo")], 10))

cat("\nXGBoost模型预测的前10个最佳组合 (带特征标签):\n")
print(head(xgb_sorted_labeled[, c(raw_data_result$feature_names, "xgb_pred", "feature_combo")], 10))

# 显示所有模型共同预测的组合（带特征标签）
if (all_models_overlap > 0) {
  common_indices <- Reduce(intersect, list(lasso_top10_indices, mlp_top10_indices, xgb_top10_indices))
  common_combinations <- combined_new_data_results[common_indices, ]
  
  cat("\n所有三个模型共同预测的前", all_models_overlap, "个最佳组合 (带特征标签):\n")
  print(common_combinations[, c(raw_data_result$feature_names, 
                                "lasso_pred", "mlp_pred", "xgb_pred",
                                "feature_combo")])
}

# 创建一个更易读的结果表格，包含特征标签和三个模型的预测值
readable_results <- combined_new_data_results[, c("feature_combo", "lasso_pred", "mlp_pred", "xgb_pred")]
readable_results <- readable_results[order(-rowMeans(readable_results[, c("lasso_pred", "mlp_pred", "xgb_pred")])), ]

cat("\n按三个模型预测平均值排序的前10个组合:\n")
print(head(readable_results, 10))

# -------- 创建预测值的统计可视化 --------

library(ggplot2)
library(dplyr)
library(tidyr)
library(viridis)
library(gridExtra)
library(scales)

features_plots_dir <- paste0(paths$feature_cache,"/plots")
dir.create(features_plots_dir, recursive = TRUE)

# 读取预测结果数据
combined_predictions <- read.csv(file.path(paths$feature_cache, "combined_predictions_with_labels.csv"))

# 生成并保存所有图表
cat("正在生成模型分布图...\n")
model_distributions(combined_predictions, config)

cat("正在生成提升度对比图...\n")
improvement_comparison(combined_predictions, config)

cat("正在生成预测散点图矩阵...\n")
prediction_scatterplot_matrix(combined_predictions, config)

cat("正在生成前10个最佳组合图...\n")
top_combinations_chart(combined_predictions, config)

cat("正在生成特征影响热图...\n")
feature_impact_heatmap(combined_predictions, config)

cat("正在生成提升度范围分布图...\n")
improvement_range_chart(combined_predictions, config)

cat("正在生成特征数量影响图...\n")
feature_count_impact(combined_predictions, config)

cat("正在生成模型一致性分析图...\n")
model_agreement_chart(combined_predictions, config)

cat("所有可视化图表已生成并保存至:", features_plots_dir, "\n")


# -------- 创建预测值的互作网络可视化 --------

# 创建一个目录用于保存互作网络图
net_dir <- file.path(paths$feature_cache, "plots")
dir.create(net_dir, showWarnings = FALSE, recursive = TRUE)

# 在运行网络生成函数时，需要确保 feature_cols 是可用的
feature_cols <- raw_data_result$feature_names # 假设 raw_data_result 中存储了特征名称

cat("\n生成模型预测的互作网络图...\n")
model_plots <- generate_model_interaction_networks(combined_new_data_results, feature_cols, threshold_diff = 0.01)

cat("\n生成共识互作网络图...\n")
consensus_plot <- generate_consensus_interaction_network(combined_new_data_results, feature_cols, threshold_diff = 13.5)

cat("\n互作网络分析完成! 结果保存至:", normalizePath(net_dir), "\n")

# 生成并显示正向互作摘要表 (此函数逻辑可能也需要根据新的互作定义进行调整，但这里先保持不变)
positive_summary <- summarize_positive_interactions(combined_new_data_results)
if (!is.null(positive_summary)) {
  cat("\n所有模型都预测有正向提升的特征组合摘要 (前10行):\n")
  print(head(positive_summary, 10))
} else {
  cat("\n没有找到所有模型都预测有正向提升的特征组合\n")
}
