import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("dark_background")
class FAPDAtaHandler():

    def __init__(self, file_name) -> None:
        
        self.data_file_name = file_name
        self.colors = ["red", "green"]
        self.labels = [["accelerations_T 120", "accelerations_S 120"], ["acceleration_T 500", "acceleration_S 500"]]

        self.acceleration_data = pd.DataFrame(columns=["S", "T", "W"], index=["75 Fa 120", "100 Fa 120", "125 Fa 120", 
                                                                              "150 Fa 120", "175 Fa 120", "200 Fa 120",
                                                                                "250 Fa 120"])


        self.fig, self.axis = plt.subplots()
    
    def load_data(self):

        data_list = []
        with open(self.data_file_name, "r") as file:

            data_from_file = file.readlines()
            for data_per_line in data_from_file:
                

                data_list_per_line = data_per_line.split()
                if "acc" in data_list_per_line:
                    
                    start_bound = data_list_per_line.index("[")
                    end_bound = data_list_per_line.index("]")
                    curent_acc_vector = [float(projection) for projection in data_list_per_line[(start_bound + 1): end_bound]]
                    data_list.append(curent_acc_vector)
        
        data_tensor = np.asarray(data_list)
        print(data_tensor)
        self.acceleration_data.iloc[:, 0] = data_tensor[:, 0]
        self.acceleration_data.iloc[:, 1] = data_tensor[:, 1]
        self.acceleration_data.iloc[:, 2] = data_tensor[:, 2]

        #print(self.acceleration_data)
    
    def show_data(self):

        self.axis.plot(self.acceleration_data.iloc[:7, 0], color="r", label="S acceleration 120")
        self.axis.plot(self.acceleration_data.iloc[:7, 1], color="g", label="T acceleration 120")
        self.axis.plot(self.acceleration_data.iloc[:7, 2], color="b", label="ABS acceleration 120")

        # self.axis[1].plot(self.acceleration_data.iloc[7:, 0], color="r", label="S acceleration 500")
        # self.axis[1].plot(self.acceleration_data.iloc[7:, 1], color="g", label="T acceleration 500")
        # self.axis[1].plot(self.acceleration_data.iloc[7:, 2], color="b", label="W acceleration 500")

        self.axis.legend(loc="upper left")
        plt.show()


if __name__ == "__main__":

    sim_ob = FAPDAtaHandler(file_name="space_mission_params_file.txt")
    sim_ob.load_data()
    sim_ob.show_data()
        

        

        

    
        