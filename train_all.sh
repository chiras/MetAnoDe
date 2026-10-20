#!/usr/bin/env bash
###############################################################################
# Train all MetAnoDe models (revision 1) with the Docker image.
# Run from the MetAnoDe base directory (contains data/ and models/):
#   bash train_all.sh                       # all models
#   bash train_all.sh fungal_ITS2 plant_rbcL  # selected models (by key, see MODELS below)
# Options via environment:
#   VERSION=m2.1  CHIMERA_MODE=homologous  EPOCHS=400  SEED=42  IMAGE=metanode:tf-25.01
#   SKIP_EXISTING=1   skip models whose models/<name>/Stats.txt already contains an Ensemble report
#   DRY_RUN=1         only print the commands
# Each model runs in its own container, one after another (each uses the whole GPU).
# Files are written with your user and group, not root.
###############################################################################
set -uo pipefail
VERSION="${VERSION:-m2.1}"
CHIMERA_MODE="${CHIMERA_MODE:-homologous}"
EPOCHS="${EPOCHS:-400}"
SEED="${SEED:-42}"
IMAGE="${IMAGE:-metanode:tf-25.01}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DRY_RUN="${DRY_RUN:-0}"
D=data

# key | model name | true file | off-target files (comma-separated) | query | length factors
MODELS=(
  "bacterial_16S-V4|bacterial_16S-V4_${VERSION}|$D/bacteria_16S-V4/16S.silva.trim.derep.fa.gz|$D/bacteria_16S-V4/16S.mitochondria.trim.2.derep.fasta.gz,$D/bacteria_16S-V4/16S.chloroplast.trim.2.derep.fasta.gz|$D/bacteria_16S-V4/16S_target.unseen.trimmed.fa|0.75,1.25"
  "fungal_ITS1|fungal_ITS1_${VERSION}|$D/fungi_ITS1/fungal_ITS1.target.final.fasta.gz|$D/fungi_ITS1/fungal_ITS1.plants.train.fasta.gz|$D/fungi_ITS1/fungal_ITS1.target.eval.fa|0.5,1.5"
  "fungal_ITS2|fungal_ITS2_${VERSION}|$D/fungi_ITS2/fungal_ITS2.target.final.fasta.gz|$D/fungi_ITS2/fungal_ITS2.plants.train.fasta.gz|$D/fungi_ITS2/fungal_ITS2.target.eval.fa|0.5,1.5"
  "plant_ITS2|plant_ITS2_${VERSION}|$D/plants_ITS2/ITS2.Quaresma2024.all.trimmedpy2.fasta.gz|$D/plants_ITS2/ITS2.fungi.trim.1.fasta.gz|$D/plants_ITS2/ITS2_plants_target.unseen.trimmed.fa|0.5,1.5"
  "plant_rbcL|plant_rbcL_${VERSION}|$D/plants_rbcL/plant_rbcL.target.train.fasta.gz|$D/plants_rbcL/plant_rbcL.bryophytes.train.fasta.gz,$D/plants_rbcL/plant_rbcL.green_algae.train.fasta.gz|$D/plants_rbcL/plant_rbcL.target.eval.fa|0.75,1.25"
  "invertebrate_COI-5P|invertebrate_COI-5P_${VERSION}|$D/animals_COI-5P/bold_COI_m1.1.invertebrates.taxonomic_sample.primer_trimmed.fasta.gz|$D/animals_COI-5P/bold_COI_m1.1.fungi.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.plants.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.protists.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.vertebrata.taxonomic_sample.primer_trimmed.fasta.gz|$D/animals_COI-5P/COI_invertebrates.target.eval.unseen.trimmed.fa|0.75,1.25"
  "metazoan_COI-5P|metazoan_COI-5P_${VERSION}|$D/animals_COI-5P/bold_COI_m1.1.metazoa.taxonomic_sample.primer_trimmed.fasta.gz|$D/animals_COI-5P/bold_COI_m1.1.fungi.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.plants.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.protists.taxonomic_sample.primer_trimmed.fasta.gz|$D/animals_COI-5P/COI_invertebrates.target.eval.unseen.trimmed.fa|0.75,1.25"
  "vertebrate_COI-5P|vertebrate_COI-5P_${VERSION}|$D/animals_COI-5P/bold_COI_m1.1.vertebrata.taxonomic_sample.primer_trimmed.fasta.gz|$D/animals_COI-5P/bold_COI_m1.1.invertebrates.offtarget_family20182.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.fungi.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.plants.taxonomic_sample.primer_trimmed.fasta.gz,$D/animals_COI-5P/bold_COI_m1.1.protists.taxonomic_sample.primer_trimmed.fasta.gz|$D/animals_COI-5P/COI_vertebrates.target.eval.unseen.trimmed.fa|0.75,1.25"
  "vertebrate_12S|vertebrate_12S_${VERSION}|$D/animals_12S/vertebrate_12S.target.train.fasta.gz|$D/animals_12S/vertebrate_12S.bacteria.train.fasta.gz,$D/animals_12S/vertebrate_12S.invertebrates.train.fasta.gz|$D/animals_12S/vertebrate_12S.target.eval.fa|0.5,1.5"
  "vertebrate_16S|vertebrate_16S_${VERSION}|$D/animals_16S/vertebrate_16S.target.train.fasta.gz|$D/animals_16S/vertebrate_16S.invertebrates.train.fasta.gz|$D/animals_16S/vertebrate_16S.target.eval.fa|0.5,1.5"
)
# note: vertebrate_16S has no separate eval file; the query above only serves the test prediction after training

mkdir -p train_logs
SUMMARY="train_logs/train_all.$(date +%Y%m%d_%H%M%S).summary.txt"
echo "=== train_all $(date '+%Y-%m-%d %H:%M') | version $VERSION | chimera $CHIMERA_MODE | epochs $EPOCHS | seed $SEED | image $IMAGE" | tee "$SUMMARY"

selected=("$@")
want() { [[ ${#selected[@]} -eq 0 ]] && return 0; local k; for k in "${selected[@]}"; do [[ "$k" == "$1" ]] && return 0; done; return 1; }

# check all input files first, so a typo does not stop the run after hours
missing=0
for m in "${MODELS[@]}"; do
  IFS='|' read -r key name db ots query lenf <<< "$m"
  want "$key" || continue
  for f in "$db" ${ots//,/ } "$query"; do
    [[ -s "$f" ]] || { echo "MISSING ($key): $f" | tee -a "$SUMMARY"; missing=1; }
  done
done
[[ $missing -eq 0 ]] || { echo "input files missing; nothing started" | tee -a "$SUMMARY"; exit 1; }

for m in "${MODELS[@]}"; do
  IFS='|' read -r key name db ots query lenf <<< "$m"
  want "$key" || continue
  if [[ "$SKIP_EXISTING" == "1" ]] && grep -q "Ensemble" "models/$name/Stats.txt" 2>/dev/null; then
    echo "skip    $name (Ensemble report exists; SKIP_EXISTING=0 to retrain)" | tee -a "$SUMMARY"; continue
  fi
  cmd=(docker run --gpus all --rm
       --user "$(id -u):$(id -g)" -e HOME=/tmp        # files owned by you, not root
       -v "$PWD":/data "$IMAGE"
       -db "$db" -ot "$ots" -p "$name" -query "$query"
       -e "$EPOCHS" -seed "$SEED"
       --len_factors "$lenf" --chimera_mode "$CHIMERA_MODE")
  echo "--- $(date '+%H:%M') $name" | tee -a "$SUMMARY"
  printf '%q ' "${cmd[@]}" | tee -a "$SUMMARY"; echo | tee -a "$SUMMARY"
  [[ "$DRY_RUN" == "1" ]] && continue
  log="train_logs/$name.$(date +%Y%m%d_%H%M%S).log"
  start=$(date +%s)
  if "${cmd[@]}" > "$log" 2>&1; then
    echo "ok      $name ($(( ($(date +%s) - start) / 60 )) min) log: $log" | tee -a "$SUMMARY"
  else
    echo "FAILED  $name (see $log)" | tee -a "$SUMMARY"
  fi
done
echo "=== done $(date '+%Y-%m-%d %H:%M')" | tee -a "$SUMMARY"
