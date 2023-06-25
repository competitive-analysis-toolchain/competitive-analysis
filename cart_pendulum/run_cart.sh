#!/bin/bash

# Change directory to the path of Hyst
cd ~/competitive-analysis/verisig-master/hyst-master/src

# Execute Hyst to obtain the product automaton
java -jar Hyst.jar -v -i ../../../cart_pendulum/generate_product/D_and_N_V6.xml -p flatten "" -t spaceex "" -o ../../../cart_pendulum/generate_product/D_and_N_V6_product.xml

# Check the exit code of Hyst
exit_code=$?

if [ $exit_code -eq 0 ]; then
  echo "Hyst executed successfully. Proceeding to modify the output XML file to remove the dummy DNN modes."

  # Modify the output XML file to retain one DNN mode
  sed -i 's/DNN1_DNN1/DNN1/g' ../../../cart_pendulum/generate_product/D_and_N_V6_product.xml

  # Change directory to the path of Verisig
  cd ~/competitive-analysis/verisig-master

  # Execute Verisig to obtain a flowstar model
  ./verisig -v -nf -of ../cart_pendulum/generate_product/D_and_N_V6_product.model -sc ../cart_pendulum/generate_product/D_and_N_V6_product.cfg -vc ../cart_pendulum/generate_product/D_and_N_V6_product.yml ../cart_pendulum/generate_product/D_and_N_V6_product.xml ../cart_pendulum/generate_product/DNN_cart_pendulum.yml

  # Check the exit code of Verisig
  exit_code=$?

  if [ $exit_code -eq 0 ]; then
    echo "Verisig executed successfully. Proceeding to flowstar."

    # Change directory to the path of flowstar
    cd ~/competitive-analysis/verisig-master/flowstar

    # Execute flowstar
    ./flowstar -p -d ../../cart_pendulum/generate_product/DNN_cart_pendulum.yml < ../../cart_pendulum/generate_product/D_and_N_V6_product.model

    # Check the exit code of flowstar
    exit_code=$?

    if [ $exit_code -eq 0 ]; then
      echo "flowstar executed successfully. Copying the flowpipes.json file."

      # Copy the flowpipes.json file
      cp outputs/flowpipes.json ~/competitive-analysis/verisig-master/FlowPipeAnalyser

      # Change directory to the path of FlowPipeAnalyser
      cd ~/competitive-analysis/verisig-master/FlowPipeAnalyser

      # Execute FlowPipeAnalyser
      python3 main.py -json flowpipes.json > testoutput.txt

      # Reformatting the counterexample for easier parsing and rename output file
      python3 reformatCE.py testoutput.txt flow_iteration0.txt

      # Move the output of FlowPipeAnalyser to the main folder for DNN retraining
      mv flow_iteration0.txt ~/competitive-analysis/cart_pendulum/dnn_training_code

      # Check the exit code of FlowPipeAnalyser
      exit_code=$?

      if [ $exit_code -eq 0 ]; then
        echo "FlowPipeAnalyser executed successfully."

        # Change directory to train DNN iteration-0
        cd ~/competitive-analysis/cart_pendulum/dnn_training_code

        # Execute command to train DNN iteration-0
        python main.py --flow-file flow_iteration0.txt --controller-file mpc_dataset_index.csv --input-variables posD thetaD dposD dthetaD --output-variables FD --amount_interval_points 4 --decimal-precision 3 --output-file retraining.csv --old-dataset mpc_dataset.csv --hyper-training-epochs 50 --hyper-training-factor 3 --NN-training-epochs 100 --output-folder retraining_output --hypertune false --hyperparameter_file hyperparameters

        # Check the exit code of train DNN iteration-0
        exit_code=$?

        if [ $exit_code -eq 0 ]; then
          echo "Train DNN iteration-0 completed successfully."
          # Rename the "retraining_output" folder to "Iteration_0"
          mv ~/competitive-analysis/cart_pendulum/dnn_training_code/retraining_output ~/competitive-analysis/cart_pendulum/dnn_training_code/Iteration_0

          # Rename and move the "test_data_set_iteration.csv" file
          mv ~/competitive-analysis/cart_pendulum/dnn_training_code/test_data_set_iteration.csv ~/competitive-analysis/cart_pendulum/dnn_training_code/Iteration_0/output_NN_training/test_data_set_iteration0.csv

          # Rename "old_data_plus_extra_data.csv" to "dataset_iteration0.csv"
          mv ~/competitive-analysis/cart_pendulum/dnn_training_code/old_data_plus_extra_data.csv ~/competitive-analysis/cart_pendulum/dnn_training_code/dataset_iteration0.csv

          # Rename "dnn_model_yaml" to "dnn_iteration0.yml"
          mv ~/competitive-analysis/cart_pendulum/dnn_training_code/Iteration_0/output_NN_training/dnn_model_yaml ~/competitive-analysis/cart_pendulum/dnn_training_code/Iteration_0/output_NN_training/dnn_iteration0.yml
        
          # Rename "retraining.csv" to "retraining_iteration0.csv"
          mv ~/competitive-analysis/cart_pendulum/dnn_training_code/retraining.csv ~/competitive-analysis/cart_pendulum/dnn_training_code/Iteration_0/output_NN_training/retraining_iteration0.csv

           # Ask user whether to continue or terminate
           read -p "Do you want to continue? (yes/no): " choice
           
           # Check user's choice
           if [ "$choice" != "yes" ]; then
                echo "Terminating the script."
                exit 0
           fi
		  
		  echo "Start CEGAR loop"
		  # CEGAR loop
		  for i in {1..20}; do
		    echo "Iteration $i"
                       
                         # Change directory to the path of flowstar
                         cd ~/competitive-analysis/verisig-master/flowstar
			
			# Execute flowstar
			./flowstar -p -d ../../cart_pendulum/dnn_training_code/Iteration_$((i-1))/output_NN_training/dnn_iteration$((i-1)).yml < ../../cart_pendulum/generate_product/D_and_N_V6_product.model
			
			# Check the exit code of flowstar inside CEGAR loop
			exit_code=$?
			
			if [ $exit_code -eq 0 ]; then
		           echo "flowstar inside CEGAR loop executed successfully. Copying the flowpipes.json file."
			  
			 # Copy the flowpipes.json file
             cp ~/competitive-analysis/verisig-master/flowstar/outputs/flowpipes.json ~/competitive-analysis/verisig-master/FlowPipeAnalyser
			 
			 # Change directory to the path of FlowPipeAnalyser
             cd ~/competitive-analysis/verisig-master/FlowPipeAnalyser
			 
			 # Execute FlowPipeAnalyser to parse counterexample file
             python3 main.py -json flowpipes.json > testoutput${i}.txt

			 # Call reformatCE.py to process the counterexample file
             python3 reformatCE.py testoutput${i}.txt flow_iteration${i}.txt
                         
                         # Move the counterexample to the dnn_training_code folder required for next retraining iteration
                         mv flow_iteration${i}.txt ~/competitive-analysis/cart_pendulum/dnn_training_code
		 
			 # Check the exit code of FlowPipeAnalyser inside CEGAR loop
                         exit_code=$?
			 
			 if [ $exit_code -eq 0 ]; then
                            echo "FlowPipeAnalyser inside CEGAR loop executed successfully."
			   
			 # Change directory to the dnn training code
                         ~/competitive-analysis/cart_pendulum/dnn_training_code
			   
			 # Execute the command for retraining
               python main.py --flow-file flow_iteration${i}.txt --controller-file mpc_dataset_index.csv --input-variables posD thetaD dposD dthetaD --output-variables FD --amount_interval_points 4 --decimal-precision 3 --output-file retraining${i}.csv --old-dataset dataset_iteration$((i-1)).csv --hyper-training-epochs 50 --hyper-training-factor 3 --NN-training-epochs 100 --output-folder retraining_output --hypertune false --hyperparameter_file hyperparameters
			   
			 # Rename the "retraining_output" folder as per iteration#
                         mv retraining_output Iteration_${i}
		       
                         # Rename the "test_data_set_iteration.csv" file as per iteration#
                         mv test_data_set_iteration.csv Iteration_${i}/output_NN_training/test_data_set_iteration${i}.csv
		       
                         # Rename "old_data_plus_extra_data.csv" as per iteration#
                         mv old_data_plus_extra_data.csv dataset_iteration${i}.csv
		       
                         # Rename "dnn_model_yaml" as per iteration#
                         mv Iteration_${i}/output_NN_training/dnn_model_yaml Iteration_${i}/output_NN_training/dnn_iteration${i}.yml
		       
                        # Rename "retraining.csv" as per iteration#
                        mv retraining${i}.csv Iteration_${i}/output_NN_training/retraining_iteration${i}.csv

                        # Ask user whether to continue or terminate
                        read -p "Do you want to continue? (yes/no): " choice

                        # Check user's choice
                        if [ "$choice" != "yes" ]; then
    	                   echo "Terminating the script."
                           exit 0
                        fi

			 else
               echo "FlowPipeAnalyser inside CEGAR loop encountered an error. Exiting the script."
               exit 1
             fi
		   else
              echo "flowstar inside CEGAR loop encountered an error. Exiting the script."
              exit 1
              fi
		  done
		   
 
          # Add commands for additional tools here
            echo "Generating boxplot"
            cd ~/competitive-analysis/cart_pendulum/dnn_training_code
            python boxplot.py

            echo "Generating lineplot"
            cd ~/competitive-analysis/cart_pendulum/dnn_training_code
            python lineplot.py    

		#Don't change from here
        else
          echo "Train DNN iteration-0 encountered an error. Exiting the script."
          exit 1
        fi

      else
        echo "FlowPipeAnalyser encountered an error. Exiting the script."
        exit 1
      fi

    else
      echo "flowstar encountered an error. Exiting the script."
      exit 1
    fi

  else
    echo "Verisig tool encountered an error. Exiting the script."
    exit 1
  fi

else
  echo "Hyst tool encountered an error. Exiting the script."
  exit 1
fi

