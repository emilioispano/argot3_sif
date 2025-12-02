fasta=$1

echo "CHECKING FASTA HEADER FORMAT..."
bad_headers=$(grep "^>" "$fasta" | grep -vP "^>[^|]+\|[^|]+\|[^|]+")
if [ -n "$bad_headers" ]; then
    echo "ERROR: Malformed FASTA headers detected:"
    echo "$bad_headers" | head -n 5
    echo "(... and possibly more)"
    exit 1
fi

echo "EXTRACTING PROTEINS LISTS"
grep ">" $fasta |cut -d"|" -f2 |sort --parallel=32 |uniq > intermediate_files/proteins_list.txt

echo "COMPUTING EMBEDDINGS FIRST"
python3 src/extract.py esm2_t33_650M_UR50D $fasta intermediate_files/torch_embeddings --repr_layers 33 --include per_tok

echo "EXTRACTING FAILED PROTEINS (if any)"
ls -1 intermediate_files/torch_embeddings | cut -d"." -f1 | sort | uniq > intermediate_files/embedded_prots.txt

n_total=$(wc -l < intermediate_files/proteins_list.txt)
n_embedded=$(wc -l < intermediate_files/embedded_prots.txt)

if [ "$n_total" -ne "$n_embedded" ]; then
    echo "WARNING: Some proteins were not embedded. Attempting re-embedding..."
    fail_uids=intermediate_files/without_embeddings.txt
    fail_fasta=intermediate_files/without_embeddings.fasta
    comm -13 intermediate_files/embedded_prots.txt intermediate_files/proteins_list.txt > $fail_uids
    python3 src/get_fastas_uniprot.py -i $fail_uids -u "$fasta" -o $fail_fasta
    python3 src/extract.py esm2_t33_650M_UR50D $fail_fasta intermediate_files/torch_embeddings --repr_layers 33 --include per_tok --nogpu
fi

python3 src/convert_to_tf.py -e intermediate_files/torch_embeddings -o intermediate_files/embeddings
rm -rf intermediate_files/torch_embeddings

#echo "COMPUTING PREDICTIONS"
pred_ont=intermediate_files/prediction_by_ontology
pred_all=output

mkdir -p $pred_ont
mkdir -p $pred_all

python3 src/predict_batch.py -l intermediate_files/proteins_list.txt -e intermediate_files/embeddings -o $pred_ont -b 16
python3 src/join.py -c $pred_ont/cco_batch.txt -m $pred_ont/mfo_batch.txt -b $pred_ont/bpo_batch.txt -o $pred_all/prediction_raw.txt
python3 src/propagate.py -i $pred_all/prediction_raw.txt -o $pred_all/prediction_propagated.txt -g intermediate_files/go/go.owl
