library(TCGAbiolinks)
library(SummarizedExperiment)
library(magrittr)
library(SNFtool)
library(igraph)
library(ggraph)
library(tidygraph)
library(impute)
library(matrixStats)



# 1. Reading all raw data
rna_se   <- readRDS("TCGA-BRCA_rna_se.rds")
mirna_se <- readRDS("TCGA-BRCA_mirna_se.rds")
meth_se  <- readRDS("TCGA-BRCA_meth_se.rds")     
cnv      <- readRDS("TCGA-BRCA_cnv_data_gene.rds")
subtypes <- readRDS("TCGA-BRCA_subtypes_info.rds")



# 2. Extracting expression matrix
rna_mat  <- assay(rna_se, "tpm_unstrand") %>% 
  {log2(. + 1)} %>% 
  as.data.frame()

mirna_df <- mirna_se                                   
rpm_cols <- grep("reads_per_million_miRNA_mapped", colnames(mirna_df))

mirna_mat <- mirna_df[, rpm_cols, drop = FALSE]        
if ("miRNA_ID" %in% colnames(mirna_df)) {
  rownames(mirna_mat) <- mirna_df$miRNA_ID
} else if ("miRNA_region" %in% colnames(mirna_df)) {
  rownames(mirna_mat) <- mirna_df$miRNA_region
} else {
  rownames(mirna_mat) <- mirna_df[, 1]
}
colnames(mirna_mat) <- gsub("reads_per_million_miRNA_mapped_", "", 
                            colnames(mirna_mat))

mirna_mat <- log2(mirna_mat + 1) %>% as.data.frame()
meth_mat <- assay(meth_se) %>% as.data.frame()


cnv_mat <- assay(cnv) %>% as.data.frame()


# 3. Patient ID and filter
filter_and_rename <- function(mat, omics_name) {
  barcodes <- colnames(mat)
  is_primary <- grepl("TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-01", barcodes)
  mat_filtered <- mat[, is_primary, drop = FALSE]
  colnames(mat_filtered) <- substr(colnames(mat_filtered), 1, 12)
  mat_filtered <- mat_filtered[, !duplicated(colnames(mat_filtered)), drop = FALSE]
  return(mat_filtered)
}

rna_clean   <- filter_and_rename(rna_mat, "mRNA")
mirna_clean <- filter_and_rename(mirna_mat, "miRNA")
meth_clean  <- filter_and_rename(meth_mat, "Methylation")
cnv_clean   <- filter_and_rename(cnv_mat, "CNV")



# 4. Take the intersection
common_patients <- Reduce(intersect, list(
  colnames(rna_clean), 
  colnames(mirna_clean),
  colnames(meth_clean),
  colnames(cnv_clean)
))
saveRDS(common_patients, file = "Common_Patients_TCGA-BRCA_4omics.rds")

cat("The number of final alignment Primary Tumor samples is ：", 
        length(common_patients), "\n")  



# 5. Impute
rna_mat_subset   <- rna_clean[, common_patients, drop = FALSE]
mirna_mat_subset <- mirna_clean[, common_patients, drop = FALSE]
meth_mat_subset  <- meth_clean[, common_patients, drop = FALSE]
# prot_mat_subset  <- prot_clean[, common_patients, drop = FALSE]
cnv_mat_subset   <- cnv_clean[, common_patients, drop = FALSE]
cat("RNA:", dim(rna_mat_subset), "\n")
cat("miRNA:", dim(mirna_mat_subset), "\n")
cat("Methylation:", dim(meth_mat_subset), "\n")
cat("CNV:", dim(cnv_mat_subset), "\n")


filter_na_rows <- function(mat, max_na_ratio = 0.5, name) {
  na_ratio <- rowMeans(is.na(mat))
  n_removed <- sum(na_ratio >= max_na_ratio)
  mat_filtered <- mat[na_ratio < max_na_ratio, , drop = FALSE]
  cat(sprintf("   [%s] delete %d  feature, remain %d \n", 
              name, n_removed, nrow(mat_filtered)))
  return(mat_filtered)
}


impute_knn_process <- function(mat, name) {
  mat <- as.matrix(mat)
  if (sum(is.na(mat)) > 0) {
    cat(sprintf("    [%s]KNN imputing...\n", name))
    capture.output({
      imputed_obj <- impute.knn(mat, k = 10, rowmax = 0.5,
                                colmax = 0.8, maxp = 1500)
    })
    mat <- imputed_obj$data
    cat(sprintf("   [%s]complete...\n", name))
  } else {
    cat(sprintf("   [%s]skip...\n", name))
  }
  return(mat)
}


meth_mat_subset  <- filter_na_rows(meth_mat_subset, 0.5, "Methylation")

rna_imputed   <- rna_mat_subset
mirna_imputed <- mirna_mat_subset
meth_imputed  <- impute_knn_process(meth_mat_subset, "Methylation")
cnv_imputed   <- cnv_mat_subset



# 6. Feature filter
mad_filter <- function(mat, n_top) {
  X <- as.matrix(mat)
  storage.mode(X) <- "numeric"
  mad_vals <- rowMads(X, na.rm = TRUE)
  mad_vals[is.na(mad_vals)] <- -Inf 
  n_actual <- min(n_top, nrow(X))
  top_idx <- order(mad_vals, decreasing = TRUE)[1:n_actual]
  return(X[top_idx, , drop = FALSE])
}

omics_final <- list(
  mRNA        = mad_filter(rna_imputed, n_top = 3000),  
  miRNA       = mad_filter(mirna_imputed, n_top = 800), 
  Methylation = mad_filter(meth_imputed, n_top = 2000), 
  CNV         = mad_filter(cnv_imputed, n_top = 2000)    
)

for(n in names(omics_final)) {
  cat(sprintf("   [%-12s] Final: %d features x %d samples\n", 
              n, nrow(omics_final[[n]]), ncol(omics_final[[n]])))
}


# 7. Label
if(exists("subtypes")) {
  clean_subtypes <- subtypes %>% distinct(patient, .keep_all = TRUE) 
  pam50_map <- setNames(clean_subtypes$BRCA_Subtype_PAM50, 
                        clean_subtypes$patient)
  final_labels <- pam50_map[colnames(omics_final$mRNA)]
  label_df <- data.frame(PatientID = colnames(omics_final$mRNA), 
                         Subtype = final_labels)
  write.csv(label_df, file.path(out_dir, "Patient_Labels.csv"), 
            row.names = FALSE)
}







