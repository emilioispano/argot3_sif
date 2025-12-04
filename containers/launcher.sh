#!/usr/bin/env bash

set -euo pipefail

# -----------------------------
# Usage function
# -----------------------------
usage() {
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -f <fasta>        Input FASTA file for DIAMOND/Argot"
    echo "  -o <outdir>       Output directory for results"
    echo "  -h                Show this help message"
    echo
    echo "Example:"
    echo "  $0 -f /path/to/targets.fasta -o /path/to/results/"
    exit 1
}

# -----------------------------
# Parse command-line options
# -----------------------------

fasta=""
results=""

while getopts ":f:o:h" opt; do
    case $opt in
        f) fasta=$OPTARG ;;
        o) results=$OPTARG ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument." >&2; usage ;;
    esac
done

# -----------------------------
# Check required arguments
# -----------------------------
if [[ -z "$fasta" || -z "$results"  ]]; then
    echo "Error: -f <fasta>, and -o <outdir> are required arguments"
    exit 1
fi

src=/argot3/src
data=$results/data
preds=$results/predictions
mkdir -p $results $data $preds

echo "CHECKING FASTA HEADER FORMAT..."
python3 $src/check_fasta.py -f $fasta -o $data/proteins_list.fasta

echo "EXTRACTING PROTEINS LISTS"
grep ">" $data/proteins_list.fasta |cut -d" " -f1 |sort --parallel=32 |uniq |sed 's/>//g' > $data/proteins_list.txt

echo "COMPUTING EMBEDDINGS FIRST"
python3 $src/extract.py esm2_t33_650M_UR50D $data/proteins_list.fasta $data/torch_embeddings --repr_layers 33 --include per_tok

echo "EXTRACTING FAILED PROTEINS (if any)"
ls -1 $data/torch_embeddings | cut -d"." -f1 | sort | uniq > $data/embedded_prots.txt

n_total=$(wc -l < $data/proteins_list.txt)
n_embedded=$(wc -l < $data/embedded_prots.txt)

if [ "$n_total" -ne "$n_embedded" ]; then
    echo "WARNING: Some proteins were not embedded. Attempting re-embedding..."
    fail_uids=$data/without_embeddings.txt
    fail_fasta=$data/without_embeddings.fasta
    comm -13 $data/embedded_prots.txt $data/proteins_list.txt > $data/without_embeddings.txt
    python3 $src/get_fastas_uniprot.py -i $data/without_embeddings.txt -u $data/proteins_list.fasta -o $data/without_embeddings.fasta
    python3 $src/check_fasta.py -f $without_embeddings.fasta -o $data/without_embeddings_clean.fasta
    python3 $src/extract.py esm2_t33_650M_UR50D $data/without_embeddings_clean.fasta $data/torch_embeddings --repr_layers 33 --include per_tok --nogpu
fi

echo "CONVERTING"
python3 $src/convert_to_tf.py -e $data/torch_embeddings -o $data/embeddings
rm -rf $data/torch_embeddings

#echo "COMPUTING PREDICTIONS"

python3 $src/predict_batch.py -l $data/proteins_list.txt -e $data/embeddings -o $data -b 16 -s /argot3/input/structure/ -w /argot3/input/weights/
python3 $src/join.py -c $data/cco_batch.txt -m $data/mfo_batch.txt -b $data/bpo_batch.txt -o $data/prediction_raw.txt
python3 $src/propagate.py -i $data/prediction_raw.txt -o $data/prediction_propagated.txt -g /argot3/input/go/go.owl
python3 $src/format_out.py -i $data/prediction_raw.txt -g /argot3/input/go/go.owl -o $preds/unpropagated.tsv
python3 $src/format_out.py -i $data/prediction_propagated.txt -g /argot3/input/go/go.owl -o $preds/propagated.tsv
