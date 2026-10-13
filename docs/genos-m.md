# Frozen Genos-m embeddings

SemiBin2 can use [Genos-m](https://github.com/BGI-HangzhouAI/Genos-m) to
encode each contig directly from its DNA sequence. This mode freezes Genos-m,
uses its pooled sequence vectors without a projection head, and skips training
SemiBin's MLP. The existing k-mer/abundance feature table is still needed:
short-read clustering intersects Genos-m and original-feature neighbor graphs,
and the normal abundance and marker-gene steps remain in use.

Install the optional Python dependencies in the SemiBin environment:

```bash
python -m pip install -e '.[genos-m]'
```

For single-sample short-read binning:

```bash
SemiBin2 single_easy_bin \
    -i contigs.fa -b reads.sorted.bam -o genos_bins \
    --embedding-source genos-m
```

The same option is accepted by `multi_easy_bin`, `bin`/`bin_short`, and
`bin_long`. For a local copy of the model, pass
`--genos-model /path/to/Genos-m`. The default model ID is
`BGI-HangzhouAI/Genos-m`. `--genos-window-size` (default 8192 bases) and
`--genos-batch-size` (default 1 window) control inference memory. Long
contigs are split into consecutive windows; token embeddings are pooled within
each window using the attention mask, then window embeddings are averaged in
proportion to their nucleotide lengths. No contig suffix is truncated.
Non-ACGTN characters in the FASTA are replaced with `N` before tokenization.

If `data.csv` has already been generated, run clustering without a SemiBin
model file:

```bash
SemiBin2 bin_short \
    -i contigs.fa --data previous_output/data.csv -o genos_bins \
    --embedding-source genos-m
```

The program automatically rescales Genos-m vectors to the distance range used
by the short-read graph weights. This scalar preserves the Genos-m neighbor
ordering. For long reads, it first sets the sequence-vector scale relative to
abundance, then calibrates the combined vector for the DBSCAN thresholds. No
part of this procedure trains Genos-m or makes its neighborhoods more
biologically accurate. Genos-m mode does not write `model.pt`, and the
existing `--model` and `--environment` options cannot be combined with it.

Genos-m is much larger than SemiBin's MLP. Its first use may download model
weights, and running it on CPU can be very slow and memory intensive. Compare
resulting bins with standard SemiBin2 before drawing conclusions about bin
quality.
