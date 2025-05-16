import matplotlib.pyplot as plt
import numpy as np
import argparse
# import pandas as pd 



# data_file = "/homes/dshudrenko/ML/ErrPs-data-augmentation/DATA/features/all_data.csv"
# data_file = "DATA/fsubject/f_sub_1.csv"
# # neg_data = "/homes/kbritt1/cs360/ErrPs-data-augmentation/DATA/was_generated_data_neg.csv"
# # pos_data = "/homes/kbritt1/cs360/ErrPs-data-augmentation/DATA/was_generated_data_pos.csv"
# neg_data = None
# pos_data = None
# parsed_data = np.loadtxt(data_file, delimiter = ',')


# pos_parsed_data = parsed_data[np.where(parsed_data[:,0] == 1)]
# pos_parsed_data= pos_parsed_data[:,3:]

# neg_parsed_data = parsed_data[np.where(parsed_data[:,0] == 0)]
# neg_parsed_data = neg_parsed_data[:,3:]

# if pos_data and neg_data:
#     pos_parsed_data = np.loadtxt(pos_data, delimiter =',')
#     neg_parsed_data = np.loadtxt(neg_data, delimiter = ',')

# print(neg_parsed_data.shape[0], pos_parsed_data.shape[0])
# pos_sensor_lists = np.zeros((10,640))

# neg_sensor_lists = np.zeros((10,640))

# cur_subject = parsed_data[0].reshape(10,640)
def plotElectrode(pos_path, neg_path=None, output="./IMAGE", title=""):
    pos_parsed_data = None
    neg_parsed_data = None
    if pos_path and neg_path  != "":
        pos_parsed_data = np.loadtxt(pos_path, delimiter =',')
        neg_parsed_data = np.loadtxt(neg_path, delimiter = ',')
    else:
        parsed_data = np.loadtxt(pos_path, delimiter = ',')
        pos_parsed_data = parsed_data[np.where(parsed_data[:,0] == 1)]
        pos_parsed_data= pos_parsed_data[:,3:]

        neg_parsed_data = parsed_data[np.where(parsed_data[:,0] == 0)]
        neg_parsed_data = neg_parsed_data[:,3:]

    pos_sensor_lists = np.zeros((10,640)) # will hold avg values

    neg_sensor_lists = np.zeros((10,640)) # wil l hold avg values
    for i in range(neg_parsed_data.shape[0]):
        neg_sensor_lists += neg_parsed_data[i].reshape(10,640)



    for i in range(pos_parsed_data.shape[0]):
        pos_sensor_lists += pos_parsed_data[i].reshape(10,640)

    

    pos_sensor_lists = pos_sensor_lists/pos_parsed_data.shape[0]
    neg_sensor_lists = neg_sensor_lists/neg_parsed_data.shape[0]
    # Loop through each sensor and its corresponding subplot

    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(22,15))

    axes = axes.flatten()
    sensor_names = ['F3','Fz','F4','FCz','C3','Cz','C4','P3','Pz','P4']
    for i in range(pos_sensor_lists.shape[0]):
        ax = axes[i]
        ax.plot(pos_sensor_lists[i], label='Incorrect Response', color='#EB5B00')
        ax.plot(neg_sensor_lists[i], label='Correct Response', color='#00008B')
        
        # Set title and labels
        ax.set_title(f'Channel {sensor_names[i]}')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Voltage Normalized (V)')
        
        # Add a legend
        ax.legend( loc='lower center')
    plt.subplots_adjust(hspace=0.4, wspace=0.4)  # Increase vertical space between subplots
    plt.suptitle(title)
    plt.savefig(output,format = 'svg')
    # for i in range(neg_sensor_lists.shape):
    # plt.plot(neg_sensor_lists[0], label = 'No Error')
    # plt.plot(pos_sensor_lists[0], label = 'Error')
    # plt.legend()
    # plt.savefig("AAAAAAAAAAA.svg", format = 'svg')
    # plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--pos_data")
    parser.add_argument("-n", "--neg_data", type=str, default ="")
    parser.add_argument("-o", "--output")
    parser.add_argument("-t","--title")
    args = parser.parse_args()
    plotElectrode(args.pos_data,args.neg_data, output = args.output)


if __name__ == "__main__":
    main()
