fasta=$1

echo "CHECKING FASTA HEADER FORMAT..."
mkdir -p /wdir/data
python3 /argot3/src/check_fasta.py -f $fasta -o /wdir/data/proteins_list.fasta

echo "EXTRACTING PROTEINS LISTS"
grep ">" /wdir/data/proteins_list.fasta |cut -d" " -f1 |sort --parallel=32 |uniq |sed 's/>//g' > /wdir/data/proteins_list.txt

echo "COMPUTING EMBEDDINGS FIRST"
python3 /argot3/src/extract.py esm2_t33_650M_UR50D /wdir/data/proteins_list.fasta /wdir/data/torch_embeddings --repr_layers 33 --include per_tok

echo "EXTRACTING FAILED PROTEINS (if any)"
ls -1 /wdir/data/torch_embeddings | cut -d"." -f1 | sort | uniq > /wdir/data/embedded_prots.txt

n_total=$(wc -l < /wdir/data/proteins_list.txt)
n_embedded=$(wc -l < /wdir/data/embedded_prots.txt)

if [ "$n_total" -ne "$n_embedded" ]; then
    echo "WARNING: Some proteins were not embedded. Attempting re-embedding..."
    fail_uids=/wdir/data/without_embeddings.txt
    fail_fasta=/wdir/data/without_embeddings.fasta
    comm -13 /wdir/data/embedded_prots.txt /wdir/data/proteins_list.txt > /wdir/data/proteins_list.fasta
    python3 /argot3/src/get_fastas_uniprot.py -i /wdir/data/proteins_list.fasta -u $fasta -o /wdir/data/without_embeddings.fasta
    python3 /argot3/src/check_fasta.py -f /wdir/data/without_embeddings.fasta -o /wdir/data/without_embeddings_clean.fasta
    python3 /argot3/extract.py esm2_t33_650M_UR50D /wdir/data/without_embeddings_clean.fasta /wdir/data/torch_embeddings --repr_layers 33 --include per_tok --nogpu
fi

echo "CONVERTING"
python3 /argot3/src/convert_to_tf.py -e /wdir/data/torch_embeddings -o /wdir/data/embeddings
rm -rf /wdir/data/torch_embeddings

#echo "COMPUTING PREDICTIONS"
pred_ont=/wdir/data/prediction_by_ontology
pred_all=/wdir/output

mkdir -p $pred_ont
mkdir -p $pred_all

python3 /argot3/src/predict_batch.py -l /wdir/data/proteins_list.txt -e /wdir/data/embeddings -o $pred_ont -b 16 -s /argot3/input/structure/ -w /argot3/input/weights/
python3 /argot3/src/join.py -c $pred_ont/cco_batch.txt -m $pred_ont/mfo_batch.txt -b $pred_ont/bpo_batch.txt -o $pred_all/prediction_raw.txt
python3 /argot3/src/propagate.py -i $pred_all/prediction_raw.txt -o $pred_all/prediction_propagated.txt -g /argot3/input/go/go.owl
python3 /argot3/src/format_out.py -i $pred_all/prediction_raw.txt -g /argot3/input/go/go.owl -o $pred_all/prediction_raw.txt.final
python3 /argot3/src/format_out.py -i $pred_all/prediction_propagated.txt -g /argot3/input/go/go.owl -o $pred_all/prediction_propagated.txt.final
mv $pred_all/prediction_raw.txt.final $pred_all/predictions_raw.tsv
mv $pred_all/prediction_propagated.txt.final $pred_all/predictions_prop.tsv
rm $pred_all/prediction_raw.txt $pred_all/prediction_propagated.txt
