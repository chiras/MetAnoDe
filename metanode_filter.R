# Filter a phyloseq object using MetAnoDe predictions, optionally combined
# with prevalence and abundance thresholds. 

#
# Why two kinds of rules
# ----------------------
# MetAnoDe classes fall into two groups that need different treatment:
#
#   error classes      substitution, indel, chimera. Typically rare across
#                      samples. A conservative rule (remove only if also rare)
#                      protects genuine taxa that were misclassified.
#   off-target classes e.g. chloroplast or mitochondrial 16S. Often ABUNDANT
#                      and PREVALENT, so a rarity rule would keep exactly
#                      these. By default they are removed regardless of
#                      abundance (remove_offtarget_always = TRUE).
#
# Note: chimeras and systematic errors from abundant parents can be recurrent
# across samples. The conservative rule will keep some of these. This is the
# accepted trade-off of the conservative mode.
#
# Removal rule for error classes (and off-target if remove_offtarget_always
# is FALSE):
#
#   remove = flagged AND low_prevalence AND low_abundance
#
# A threshold set to NULL means that condition is not applied (always TRUE):
#
#   both thresholds given        flagged AND low_prev AND low_abund
#   only abundance_threshold     flagged AND low_abund
#   only prevalence_threshold    flagged AND low_prev
#   both NULL                    all flagged taxa removed
#
# Abundance metric
# ----------------
# Median of non-zero values by default. The plain median would be zero for
# real but patchy taxa and would mark them as low-abundance.
#
# Data type
# ---------
# Detected from sample sums (empty samples ignored):
#   relative  sums ~ 1      abundance_threshold should be < 1
#   percent   sums ~ 100    abundance_threshold in percent (e.g. 1 = 1%)
#   counts    integers      abundance_threshold in reads (e.g. >= 1)
# The threshold must be in the same unit as the OTU table.
#
# ID matching
# -----------
# Taxa absent from the MetAnoDe table cannot be judged and are KEPT. A warning
# is given, because a systematic name mismatch (e.g. ";size=123" suffixes)
# would otherwise make the filter silently ineffective. strip_size = TRUE
# removes such suffixes on both sides before matching.
#
# Output
# ------
# Filtered phyloseq object with attributes:
#   "metanode_filter_summary"  one row per taxon: class, class group,
#                              prevalence, abundance, decision, reason
#   "metanode_filter_params"   parameters used (for provenance)
#
# Examples
# --------
# library(phyloseq)
# source("metanode_filter.R")
#
# # Conservative for error classes, off-target always removed (recommended)
# ps_filt <- metanode_filter(ps, "predictions/run_Ensemble.query.csv",
#                            abundance_threshold = 0.01,
#                            prevalence_threshold = 0.02,
#                            offtarget_labels = c("chloroplast", "mitochondria"))
#
# # Remove all flagged taxa
# ps_filt <- metanode_filter(ps, meta,
#                            abundance_threshold = NULL,
#                            prevalence_threshold = NULL,
#                            offtarget_labels = c("chloroplast", "mitochondria"))
#
# head(attr(ps_filt, "metanode_filter_summary"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

.mnf_strip_size <- function(x) {
  # remove ";size=123" or ";size=123;" (VSEARCH/USEARCH abundance annotation)
  sub(";?size=[0-9]+;?$", "", x)
}

.mnf_check_threshold <- function(x, name, upper = Inf) {
  if (is.null(x)) return(invisible(NULL))
  if (!is.numeric(x) || length(x) != 1 || is.na(x) || x < 0 || x > upper) {
    stop("`", name, "` must be NULL or a single number between 0 and ", upper, ".")
  }
}

# Detect data type from sample sums. Empty samples are ignored, because
# their sum of 0 says nothing about the scale.
.mnf_detect_type <- function(otu_mat, rel_tol) {
  sums <- colSums(otu_mat, na.rm = TRUE)
  sums <- sums[sums > 0]
  if (length(sums) == 0) stop("All samples are empty.")
  if (all(abs(sums - 1) < rel_tol)) return("relative")
  if (all(abs(sums - 100) < rel_tol * 100)) return("percent")
  if (all(abs(otu_mat - round(otu_mat)) < 1e-9, na.rm = TRUE)) return("counts")
  "unknown"
}

.mnf_check_units <- function(type, abundance_threshold) {
  if (is.null(abundance_threshold)) return(invisible(NULL))
  if (type == "relative" && abundance_threshold >= 1) {
    warning("Table is relative abundance (sums ~1), but abundance_threshold >= 1. ",
            "No taxon can pass this threshold.")
  }
  if (type == "percent" && abundance_threshold > 100) {
    warning("Table is in percent (sums ~100), but abundance_threshold > 100.")
  }
  if (type == "counts" && abundance_threshold < 1) {
    warning("Table contains counts, but abundance_threshold < 1. ",
            "It is effectively a presence check.")
  }
  if (type == "unknown") {
    warning("Could not determine data type (not counts, relative or percent). ",
            "Check that abundance_threshold matches the table units.")
  }
}


# ---------------------------------------------------------------------------
# Core decision (pure function on a taxa x samples matrix; testable without
# phyloseq)
# ---------------------------------------------------------------------------

metanode_filter_decide <- function(
    otu_mat,                       # taxa in rows, samples in columns
    class_vec,                     # named character vector: names = taxa IDs
    abundance_threshold = 0.01,
    prevalence_threshold = 0.02,
    true_label = "EN_positive",
    offtarget_labels = NULL,
    remove_offtarget_always = TRUE,
    use_nonzero_median = TRUE,
    rel_tol = 1e-6,
    strip_size = FALSE,
    verbose = TRUE
) {
  .mnf_check_threshold(abundance_threshold, "abundance_threshold")
  .mnf_check_threshold(prevalence_threshold, "prevalence_threshold", upper = 1)

  taxa_ids <- rownames(otu_mat)
  if (is.null(taxa_ids)) stop("OTU table must have taxa names.")

  # --- ID matching -------------------------------------------------------
  key_taxa <- if (strip_size) .mnf_strip_size(taxa_ids) else taxa_ids
  key_meta <- if (strip_size) .mnf_strip_size(names(class_vec)) else names(class_vec)

  # NA predictions cannot be used; treat as unmatched
  ok <- !is.na(class_vec) & class_vec != ""
  class_vec <- class_vec[ok]; key_meta <- key_meta[ok]

  dup <- unique(key_meta[duplicated(key_meta)])
  if (length(dup) > 0) {
    stop("Duplicated IDs in MetAnoDe table (", length(dup), "), e.g. ",
         paste(head(dup, 3), collapse = ", "), ".")
  }
  names(class_vec) <- key_meta

  matched <- key_taxa %in% key_meta
  if (!all(matched)) {
    warning(sum(!matched), " of ", length(taxa_ids),
            " taxa have no MetAnoDe prediction and are kept unfiltered. Examples: ",
            paste(head(taxa_ids[!matched], 3), collapse = ", "),
            if (!strip_size) ". If IDs carry ';size=' suffixes, use strip_size = TRUE." else ".")
  }

  pred <- rep(NA_character_, length(taxa_ids))
  pred[matched] <- unname(class_vec[key_taxa[matched]])

  # --- class groups ------------------------------------------------------
  observed <- unique(pred[!is.na(pred)])
  if (!true_label %in% observed) {
    warning("true_label '", true_label, "' does not occur in the predictions. ",
            "Check `true_label`; otherwise every matched taxon counts as flagged.")
  }
  missing_ot <- setdiff(offtarget_labels, observed)
  if (length(missing_ot) > 0 && verbose) {
    message("Off-target labels not present in predictions: ",
            paste(missing_ot, collapse = ", "))
  }
  if (is.null(offtarget_labels) && remove_offtarget_always) {
    warning("No offtarget_labels given: all non-true classes are treated as ",
            "error classes, so off-target taxa are only removed if rare. Labels found: ",
            paste(observed, collapse = ", "))
  }

  group <- ifelse(is.na(pred), "unmatched",
           ifelse(pred == true_label, "true",
           ifelse(pred %in% offtarget_labels, "offtarget", "error")))

  # --- prevalence and abundance -----------------------------------------
  type <- .mnf_detect_type(otu_mat, rel_tol)
  .mnf_check_units(type, abundance_threshold)

  n_empty <- sum(colSums(otu_mat, na.rm = TRUE) == 0)
  if (n_empty > 0) {
    warning(n_empty, " empty sample(s) found. They lower the prevalence of every ",
            "taxon; consider removing them first.")
  }

  prevalence <- rowMeans(otu_mat > 0, na.rm = TRUE)

  abundance_value <- apply(otu_mat, 1, function(x) {
    if (use_nonzero_median) {
      x <- x[!is.na(x) & x > 0]
      if (length(x) == 0) return(0)
    }
    stats::median(x, na.rm = TRUE)
  })

  # NULL threshold = condition not applied (TRUE for all taxa)
  low_prev  <- if (is.null(prevalence_threshold)) rep(TRUE, length(taxa_ids))
               else prevalence < prevalence_threshold
  low_abund <- if (is.null(abundance_threshold)) rep(TRUE, length(taxa_ids))
               else abundance_value < abundance_threshold
  rare <- low_prev & low_abund

  # --- removal decision --------------------------------------------------
  remove_offtarget <- group == "offtarget" &
    (remove_offtarget_always | rare)
  remove_error <- group == "error" & rare
  remove <- remove_offtarget | remove_error

  reason <- ifelse(group == "unmatched", "kept: no prediction",
            ifelse(group == "true", "kept: true",
            ifelse(remove_offtarget & remove_offtarget_always, "removed: off-target",
            ifelse(remove, "removed: flagged and rare",
                   "kept: flagged but not rare"))))

  summary_df <- data.frame(
    taxon = taxa_ids,
    metanode_class = pred,
    class_group = group,
    prevalence = prevalence,
    abundance_value = abundance_value,
    low_prevalence = low_prev,
    low_abundance = low_abund,
    removed = remove,
    reason = reason,
    stringsAsFactors = FALSE,
    row.names = NULL
  )

  params <- list(
    abundance_threshold = abundance_threshold,
    prevalence_threshold = prevalence_threshold,
    true_label = true_label,
    offtarget_labels = offtarget_labels,
    remove_offtarget_always = remove_offtarget_always,
    use_nonzero_median = use_nonzero_median,
    strip_size = strip_size,
    detected_type = type
  )

  if (verbose) {
    message("Detected table type: ", type)
    message("Abundance threshold: ",
            if (is.null(abundance_threshold)) "not applied" else abundance_threshold)
    message("Prevalence threshold: ",
            if (is.null(prevalence_threshold)) "not applied" else prevalence_threshold)
    message("Matched taxa: ", sum(matched), " / ", length(taxa_ids))
    print(table(reason))
  }

  list(keep = stats::setNames(!remove, taxa_ids), summary = summary_df, params = params)
}


# ---------------------------------------------------------------------------
# phyloseq wrapper
# ---------------------------------------------------------------------------

metanode_filter <- function(
    phyloseq,
    metanode,                      # data.frame or path to MetAnoDe CSV
    abundance_threshold = 0.01,
    prevalence_threshold = 0.02,
    class_col = "EN_class",
    true_label = "EN_positive",
    id_col = "headers",
    offtarget_labels = NULL,       # MetAnoDe labels of off-target classes
    remove_offtarget_always = TRUE,
    use_nonzero_median = TRUE,
    rel_tol = 1e-6,
    strip_size = FALSE,
    verbose = TRUE
) {
  if (!inherits(phyloseq, "phyloseq")) stop("`phyloseq` must be a phyloseq object.")

  if (is.character(metanode) && length(metanode) == 1) {
    if (!file.exists(metanode)) stop("MetAnoDe file not found: ", metanode)
    meta <- utils::read.csv(metanode, stringsAsFactors = FALSE, check.names = FALSE)
  } else if (is.data.frame(metanode)) {
    meta <- metanode
  } else {
    stop("`metanode` must be a data.frame or a path to a CSV file.")
  }
  for (col in c(id_col, class_col)) {
    if (!col %in% colnames(meta)) stop("Column `", col, "` not found in MetAnoDe table.")
  }

  otu <- phyloseq::otu_table(phyloseq)
  otu_mat <- as(otu, "matrix")
  if (!phyloseq::taxa_are_rows(otu)) otu_mat <- t(otu_mat)

  class_vec <- stats::setNames(as.character(meta[[class_col]]),
                               as.character(meta[[id_col]]))

  res <- metanode_filter_decide(
    otu_mat, class_vec,
    abundance_threshold = abundance_threshold,
    prevalence_threshold = prevalence_threshold,
    true_label = true_label,
    offtarget_labels = offtarget_labels,
    remove_offtarget_always = remove_offtarget_always,
    use_nonzero_median = use_nonzero_median,
    rel_tol = rel_tol,
    strip_size = strip_size,
    verbose = verbose
  )

  ps_filt <- phyloseq::prune_taxa(res$keep, phyloseq)
  attr(ps_filt, "metanode_filter_summary") <- res$summary
  attr(ps_filt, "metanode_filter_params") <- res$params
  ps_filt
}
