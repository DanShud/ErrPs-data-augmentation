all:
	make train_gan
	make generate_data
	make plot_generated
	make plot_real


generate_data:
	python3 generate_data.py -p models/Waspos_out_1/generator.keras -n models/Wasneg_out_1/generator.keras -o ./DATA/generated_data/


train_gan: 
	python3.py -d ./DATA/features -o models/Wassneg_out_1/ -n 
	python3.py -d ./DATA/features -o models/Wasspos_out_1/ -n -p

plot_generated: 
	python3.py plotting.py -d ./DATA/generate_data/generated_data_pos.csv -n generate_data_neg.csv -o generated_plot.svg

plot_real:
	python3.py plotting.py -d ./DATA/features/all_data -o generated_plot.svg


