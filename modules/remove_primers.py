#python remove_primers.py --input input.fasta --output output.fasta --fw ATGCGATACTTGGTGTGAAT --rv TCCTCCGCTTATTGATATGC # ITS2

import argparse
import re
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Data import IUPACData

def parse_args():
    parser = argparse.ArgumentParser(description="Remove primers from sequences in a FASTA file.")
    parser.add_argument('--input', required=True, help='Input FASTA file')
    parser.add_argument('--output', required=True, help='Output FASTA file')
    parser.add_argument('--fw', help='Forward primer sequence')
    parser.add_argument('--rv', help='Reverse primer sequence (complement will be removed)')
    return parser.parse_args()

def iupac_to_regex(iupac_seq):
    iupac_dict = IUPACData.ambiguous_dna_values
    regex_seq = ''.join([f"[{iupac_dict[nuc]}]" for nuc in iupac_seq])
    return regex_seq

def remove_primers(seq, fw_primer=None, rv_primer=None):
    trimmed = {'start': False, 'end': False}
    seq_str = str(seq)  # Convert to string if it's a Seq object
    
    if fw_primer:
        fw_regex = re.compile(f"^{iupac_to_regex(fw_primer)}", re.IGNORECASE)
        if fw_regex.match(seq_str):
            seq_str = re.sub(fw_regex, "", seq_str)
            trimmed['start'] = True

    if rv_primer:
        rv_seq = str(Seq(rv_primer).reverse_complement())
        rv_regex = re.compile(f"{iupac_to_regex(rv_seq)}$", re.IGNORECASE)
        if rv_regex.search(seq_str):
            seq_str = re.sub(rv_regex, "", seq_str)
            trimmed['end'] = True

    return Seq(seq_str), trimmed

def main():
    args = parse_args()

    if not args.fw and not args.rv:
        raise ValueError("At least one of --fw or --rv must be provided.")

    trimmed_count = 0
    untrimmed_count = 0

    records = []
    for record in SeqIO.parse(args.input, "fasta"):
        new_seq, trimmed = remove_primers(record.seq, args.fw, args.rv)
        if trimmed['start'] or trimmed['end']:
            trimmed_count += 1
        else:
            untrimmed_count += 1
        new_record = SeqRecord(new_seq, id=record.id, description=record.description)
        records.append(new_record)

    SeqIO.write(records, args.output, "fasta")

    print(f"Number of sequences trimmed: {trimmed_count}")
    print(f"Number of sequences not trimmed: {untrimmed_count}")

if __name__ == "__main__":
    main()
