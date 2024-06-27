#!/bin/bash

# Define the two arrays of floating numbers
lrs=(0.05 0.06 0.07 0.08 0.09 0.1 0.2 0.3 0.4 0.5)
max_depths=(3 4 5 6 7 8 9 10)

# Loop over each element in array1
for lr in "${lrs[@]}"; do
	    # Loop over each element in array2
	        for max_depth in "${max_depths[@]}"; do
			        # Run the command with the current elements as arguments
				        # The command here is 'example_command'
					        # '$i' is the current element from array1
						        # '$j' is the current element from array2
							        # The second argument includes both 'i' and 'j'
								        python -W ignore TrainBDT.py --data-path /ceph/cms/store/user/cmantill/bbbb/skimmer/24May24_v12_private_signal/ --model-name 24Jun04_opt_1_lr_"${lr}"_max_depth_"${max_depth}" --config-name 24May31_legacy_vbf_vars --xbb bbFatJetPNetTXbbLegacy --mass bbFatJetPNetMassLegacy --legacy --sig-keys hh4b vbfhh4b-k2v0 --no-pnet-plots --apply-cuts --year 2022 2022EE 2023 2023BPix --learning-rate "$lr" --max-depth "$max_depth"
									    done
								    done
