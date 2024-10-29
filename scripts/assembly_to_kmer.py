
import os
import sys
import logging

from ai4water.utils.utils import dateandtime_now

from scripts.utils import count_kmers_in_file, aggregate_kmer_counts, write_aggregated_kmers_to_files


time_of_this_file = dateandtime_now()

jobid = sys.argv[1]

logging.basicConfig(filename=f'log_files/data/assembly_tochunk_{time_of_this_file}_{jobid}.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

logger.info("starting")

base_path = f'/ibex/project/c2205/jiayi/Kp_all/assemblies/'
logger.info("1")
k = 54
num_files = 2000  # Number of output files
output_prefix = '/ibex/project/c2205/sara/klebsiella_pneumoniae/chunk_files'
logger.info("2")

kmer_counts_list = []

logger.info("3")

file_count = 1

for file in os.listdir(base_path):
    logger.info(f"{file_count}")
    file_path = os.path.join(base_path, file)
    if os.path.exists(file_path):
        kmer_counts = count_kmers_in_file(file_path, k, file)
        kmer_counts_list.append(kmer_counts)
    file_count+=1

# for root, dirs, files in os.walk(base_path):
#     logger.info(f"{len(files)}")
#     for dir_name in dirs:
#         logger.info(f'{dir_name}')
#         sample_path = os.path.join(root, dir_name, 'assembly_unicycler', 'assembly.fasta')
#         if os.path.exists(sample_path):
#             sample_name = dir_name
#             kmer_counts = count_kmers_in_file(sample_path, k, sample_name)
#             kmer_counts_list.append(kmer_counts)

aggregated_counts = aggregate_kmer_counts(kmer_counts_list)
logger.info("4")
write_aggregated_kmers_to_files(aggregated_counts, output_prefix)
# write_aggregated_kmers_to_files(aggregated_counts, output_prefix, num_files)
logger.info("done")