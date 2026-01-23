library(TCGAbiolinks)



# 1. Extract the number of subtypes

project_id <- "TCGA-BRCA"
tumor_short <- tolower(substr(project_id, 6, 10))
subtypes_info <- TCGAquery_subtype(tumor = tumor_short)

cat("The number of unique subtypes for Project", project_id, "is:", 
    length(unique(subtypes_info$BRCA_Subtype_PAM50)), "\n")

print(table(subtypes_info$BRCA_Subtype_PAM50))
saveRDS(subtypes_info, file = paste0(project_id , "_subtypes_info.rds"))


# 2. Download clinical data
query_clinical <- GDCquery(
  project = project_id,
  data.category = "Clinical",
  data.type = "Clinical Supplement"
)
GDCdownload(query_clinical, method = "api", 
            files.per.chunk = 1, directory = "GDC_Clinical")

ids_to_exclude <- c(
  
)

query_df <- query_clinical$results[[1]]
query_df_filtered <- subset(query_df, !(id %in% ids_to_exclude))
query_clinical$results[[1]] <- query_df_filtered

clinical_data <- GDCprepare_clinic(query_clinical, directory = "GDC_Clinical",  
                 clinical.info = "patient")

saveRDS(clinical_data, file = paste0(project_id , "_clinical_data.rds"))



# 3. Download mRNA data
query_rna <- GDCquery(
  project = project_id,
  data.category = "Transcriptome Profiling",
  data.type = "Gene Expression Quantification",
  workflow.type = "STAR - Counts",       
  sample.type = "Primary Tumor"          
)
GDCdownload(query_rna, method = "api", 
            files.per.chunk = 1, directory = "GDC_mRNA")

rna_se <- GDCprepare(query_rna, directory = "GDC_mRNA")

saveRDS(rna_se, file = paste0(project_id , "_rna_se.rds"))



# 4. Download miRNA data
query_mirna <- GDCquery(
  project = project_id,
  data.category = "Transcriptome Profiling",
  data.type = "miRNA Expression Quantification", 
  sample.type = "Primary Tumor"                 
)
GDCdownload(query_mirna, method = "api", 
            files.per.chunk = 1, directory = "GDC_miRNA")
mirna_se <- GDCprepare(query_mirna, directory = "GDC_miRNA")
saveRDS(mirna_se, file = paste0(project_id , "_mirna_se.rds"))



# 5. Download DNA Methylation data
query_meth <- GDCquery(
  project = project_id,
  data.category = "DNA Methylation",
  data.type = "Methylation Beta Value",
  platform = "Illumina Human Methylation 450", 
  sample.type = "Primary Tumor"
)
GDCdownload(query_meth, method = "api", 
            files.per.chunk = 1, directory = "GDC_methylation")
meth_se <- GDCprepare(query_meth, directory = "GDC_methylation")
saveRDS(meth_se, file = paste0(project_id , "_meth_se.rds"))



# 6. Download CNV data
query_cnv <- GDCquery(
  project = project_id,
  data.category = "Copy Number Variation",
  data.type = "Gene Level Copy Number",   
  sample.type = "Primary Tumor"
)
GDCdownload(query_cnv, method = "api", 
            files.per.chunk = 1, directory = "cnv")
cnv_data_gene <- GDCprepare(query_cnv, directory = "cnv")
saveRDS(cnv_data_gene, file = paste0(project_id , "_cnv_data_gene.rds"))

