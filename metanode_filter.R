# metanode_filter.R
#
# Filter a phyloseq object using MetAnoDe anomaly predictions, optionally
# combined with abundance and prevalence thresholds.
#
# Overview
# --------
# `metanode_filter()` removes taxa that are flagged by MetAnoDe as anomalous,
# while allowing conservative retention of taxa that are sufficiently abundant
# or prevalent across samples.
#
# This is intended as an ASV-level filtering step. It differs from classical
# per-sample low-abundance filtering, which targets low-frequency observations
# within individual samples. Here, the goal is instead to decide whether a
# putative taxon should be retained at all.
#
# Filtering logic
# ---------------
# Taxa predicted by MetAnoDe as "not true" are removed only if they satisfy
# both of the following:
#
#   1. low prevalence across samples
#   2. low abundance across samples
#
# In other words, removal uses an AND rule:
#
#   remove = flagged_as_not_true AND low_prevalence AND low_abundance
#
# This is intentionally conservative. It aims to preferentially remove likely
# artefactual taxa that are both rare and weak, while reducing the risk of
# discarding biologically relevant taxa that are recurrent or abundant.
#
# Abundance metric
# ----------------
# The function uses the median of non-zero abundances by default, rather than
# the median across all samples. This avoids penalizing real but patchy taxa,
# which would otherwise often have a median of zero.
#
# Data type handling
# ------------------
# The function inspects the phyloseq OTU table and checks whether it appears to
# contain:
#
#   - raw counts   (sample sums not approximately 1)
#   - relative abundances (sample sums approximately 1)
#
# Thresholds should be chosen accordingly:
#
#   - raw counts: abundance_threshold >= 1 is typically appropriate
#   - relative abundances: abundance_threshold < 1 is typically appropriate
#
# Inputs
# ------
# phyloseq : phyloseq object
#   A phyloseq object containing an OTU table.
#
# metanode : data.frame or file path
#   MetAnoDe classification table, either already loaded as a data.frame or
#   provided as a path to a CSV file.
#
# abundance_threshold : numeric
#   Threshold below which a flagged taxon is considered low-abundance.
#
# prevalence_threshold : numeric
#   Minimum fraction of samples in which a taxon must occur to avoid being
#   treated as low-prevalence.
#
# class_col : character
#   Column in the MetAnoDe table containing class predictions.
#
# true_label : character
#   Label identifying taxa considered "true" by MetAnoDe.
#
# id_col : character
#   Column in the MetAnoDe table used to match taxa IDs to the phyloseq OTU
#   table.
#
# use_nonzero_median : logical
#   If TRUE, compute abundance from the median of non-zero values only.
#
# Output
# ------
# Returns a filtered phyloseq object. A filtering summary table is attached as:
#
#   attr(result, "metanode_filter_summary")
#
# Example
# -------
# library(phyloseq)
#
# # Example with relative abundance data
# ps_filt <- metanode_filter(
#   phyloseq = ps_rel,
#   metanode = "predictions/sample_Ensemble.query.csv",
#   abundance_threshold = 0.01,
#   prevalence_threshold = 0.02,
#   class_col = "EN_class",
#   true_label = "EN_positive",
#   id_col = "headers"
# )
#
# # Inspect filtering summary
# filt_summary <- attr(ps_filt, "metanode_filter_summary")
# head(filt_summary)
#
# # Example with raw count data
# ps_filt_counts <- metanode_filter(
#   phyloseq = ps_counts,
#   metanode = "predictions/sample_Ensemble.query.csv",
#   abundance_threshold = 10,
#   prevalence_threshold = 0.02,
#   class_col = "EN_class",
#   true_label = "EN_positive",
#   id_col = "headers"
# )
#
# Notes
# -----
# - This function is designed for ASV-level anomaly filtering.
# - It is usually best applied before or alongside standard sample-wise
#   low-abundance filtering, not as a replacement for it.
# - Off-target taxa may behave differently from PCR or sequencing artefacts,
#   so users may later wish to implement class-specific filtering rules.

metanode_filter <- function(
    phyloseq,
    metanode,
    abundance_threshold = 0.01,
    prevalence_threshold = 0.05,
    class_col = "EN_class",
    true_label = "EN_positive",
    id_col = "headers",
    use_nonzero_median = TRUE,
    rel_tol = 1e-6,
    verbose = TRUE
) {
  if (!inherits(phyloseq, "phyloseq")) {
    stop("`phyloseq` must be a phyloseq object.")
  }
  
  if (is.character(metanode) && length(metanode) == 1) {
    if (!file.exists(metanode)) {
      stop("MetAnoDe file not found: ", metanode)
    }
    meta <- utils::read.csv(metanode, stringsAsFactors = FALSE, check.names = FALSE)
  } else if (is.data.frame(metanode)) {
    meta <- metanode
  } else {
    stop("`metanode` must be a data.frame or a path to a CSV file.")
  }
  
  if (!id_col %in% colnames(meta)) {
    stop("Column `", id_col, "` not found in MetAnoDe table.")
  }
  if (!class_col %in% colnames(meta)) {
    stop("Column `", class_col, "` not found in MetAnoDe table.")
  }
  
  otu <- phyloseq::otu_table(phyloseq)
  otu_mat <- as(otu, "matrix")
  if (!phyloseq::taxa_are_rows(otu)) {
    otu_mat <- t(otu_mat)
  }
  
  taxa_ids <- rownames(otu_mat)
  if (is.null(taxa_ids)) {
    stop("OTU table must have taxa names / rownames.")
  }
  
  sample_totals <- colSums(otu_mat, na.rm = TRUE)
  is_relative <- all(abs(sample_totals - 1) < rel_tol)
  
  if (verbose) {
    message("Detected OTU table type: ", if (is_relative) "relative abundance" else "raw counts")
    message("Abundance threshold: ", abundance_threshold)
    message("Prevalence threshold: ", prevalence_threshold)
  }
  
  if (abundance_threshold < 1 && !is_relative) {
    warning("abundance_threshold < 1 but OTU table appears to be raw counts.")
  }
  if (abundance_threshold >= 1 && is_relative) {
    warning("abundance_threshold >= 1 but OTU table appears to be relative abundance.")
  }
  
  class_vec <- as.character(meta[[class_col]])
  names(class_vec) <- as.character(meta[[id_col]])
  
  matched <- taxa_ids %in% names(class_vec)
  
  flagged_not_true <- rep(FALSE, length(taxa_ids))
  names(flagged_not_true) <- taxa_ids
  flagged_not_true[matched] <- class_vec[taxa_ids[matched]] != true_label
  
  prevalence <- rowMeans(otu_mat > 0, na.rm = TRUE)
  
  abundance_stat <- function(x) {
    if (use_nonzero_median) {
      x <- x[x > 0]
      if (length(x) == 0) return(0)
    }
    stats::median(x, na.rm = TRUE)
  }
  
  abundance_value <- apply(otu_mat, 1, abundance_stat)
  
  # Removal rule: flagged AND low prevalence AND low abundance
  remove_taxa <- flagged_not_true &
    (prevalence < prevalence_threshold) &
    (abundance_value < abundance_threshold)
  
  keep_taxa <- !remove_taxa
  
  if (verbose) {
    message("Matched taxa in MetAnoDe table: ", sum(matched), " / ", length(taxa_ids))
    message("Flagged as not true: ", sum(flagged_not_true))
    message("Removed after prevalence + abundance filter: ", sum(remove_taxa))
    message("Kept taxa: ", sum(keep_taxa), " / ", length(keep_taxa))
  }
  
  phyloseq_filtered <- phyloseq::prune_taxa(keep_taxa, phyloseq)
  
  summary_df <- data.frame(
    taxon = taxa_ids,
    metanode_class = ifelse(taxa_ids %in% names(class_vec), class_vec[taxa_ids], NA),
    flagged_not_true = flagged_not_true,
    prevalence = prevalence,
    abundance_value = abundance_value,
    removed = remove_taxa,
    stringsAsFactors = FALSE
  )
  
  attr(phyloseq_filtered, "metanode_filter_summary") <- summary_df
  phyloseq_filtered
}

