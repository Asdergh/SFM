import numpy as np
import matplotlib.pyplot as plt
import os
import math as mt



class MechanicCalculator():


    def __init__(self) -> None:
        
        self.ka_mass = 1250.
        self.ka_sa = 1.5 * 1.8
        self.Cx_coeff = (2.0 / 2.5)
        self.sigma_coeff = (self.Cx_coeff * self.ka_sa) / (2 * self.ka_mass)
        
        self.h_peregee = 350.0 + 6371.0
        self.h_apogee = 450.0 + 6371.0
        self.i_param = (20.0 * np.pi) / 180.0
        self.Omega_param = (10.0 * np.pi) / 180.0
        self.omega_param = 0.0
        self.M_param = (45.0 * np.pi) / 180.0
        
        self.Sm_axis = (self.h_peregee + self.h_apogee) / 2
        self.Ext_param = (self.h_apogee - self.h_peregee) / (self.h_apogee + self.h_peregee)
        self.Focal_param = self.Sm_axis * (1 - self.Ext_param ** 2)
        
        self.earth_grav_param = 398600.4415
        self.earth_anguler_vel = 7.2921159e-5
        self.earth_ext_param = 0.0067385254
        self.static_density = 1.58868e-8
        self.SM_earth_axis = 6378136.0

        self.accuracy = (0.001 * np.pi) / 180.0
        self.ND_upper_120 = 1.58868e-8

        self.night_density_coeffs_120 = np.array([
            [26.8629, 27.4598, 28.6395, 29.6418, 30.1671, 29.7578, 30.7854],
            [-0.451674, -0.4663668, -0.490987, -0.514957, -0.527837, -0.5179115, -0.545695],
            [0.00290397, 0.002974, 0.00320649, 0.00341926, 0.00353211, 0.00342699, 0.00370328],
            [-1.069535e-5, -1.0753e-5, -1.1681e-5, -1.25785e-5, -1.30227e-5, -1.24137e-5, -1.37072e-5],
            [2.21598e-8, 2.17059e-8, 2.36847e-8, 2.5727e-8, 2.66455e-8, 2.48209e-8, 2.80614e-8],
            [-2.42941e-11, -2.30249e-11, -2.51809e-11, -2.75874e-11, -2.85432e-11, -2.58413e-11, 3.00184e-11],
            [1.09926e-14, 1.00123e-14, 1.09536e-14, 1.21091e-14, 1.25009e-14, 1.09383e-14, 1.31142e-14]
        ])

        self.night_density_coeffs_500 = np.array([
            [17.8781, -2.54909, -13.9599, -23.3079, -14.7264, -4.912, -5.40952],
            [-0.132025, 0.0140064, 0.0844951, 0.135141, 0.0713256, 0.0108326, 0.00550749],
            [0.000227717, -0.00016946, -0.000328875, -0.000420802, -0.000228015, -8.10546e-5, -3.78851e-5],
            [-2.2543e-7, 3.27196e-7, 5.05918e-7, 5.73717e-7, 2.8487e-7, 1.15712e-7, 2.4808e-8],
            [1.33574e-10, -2.8763e-10, -3.92299e-10, -4.03238e-10, -1.74383e-10, -8.13296e-11, 4.92183e-12],
            [-4.50458e-14, 1.22625e-13, 1.52279e-13, 1.42846e-13, 5.08071e-14, 3.04913e-14, -8.65011e-15],
            [6.72086e-18, -2.05736e-17, -2.35576e-17, -2.01726e-17, -5.34955e-18, -4.94989e-18, 1.9849e-18]
        ])

        self.night_density_120 = np.zeros(6)
        self.night_density_500 = np.zeros(6)

        print(self.night_density_coeffs_120.shape)
        print(self.night_density_coeffs_500.shape)
        self.constant_night_density = 1.58868e-8


    def _calculate_anomalies(self):

        self.ext_anomaly_past = self.M_param
        self.ext_anomaly = self.M_param + self.Ext_param * np.sin(self.ext_anomaly_past)

        while (self.ext_anomaly - self.ext_anomaly_past) > self.accuracy:

            self.ext_anomaly_past = self.ext_anomaly
            self.ext_anomaly = self.M_param + self.Ext_param * np.sin(self.ext_anomaly_past)

        self.theta_anomaly = 2 * np.arctanh(np.sqrt((1 + self.Ext_param) / (1 - self.Ext_param)) * np.tan(self.ext_anomaly / 2))
        self.position_norma_AGSK = self.Focal_param * (1 - self.Ext_param ** 2) / (1 + self.Ext_param * np.cos(self.theta_anomaly))
        self.u_param = self.theta_anomaly + self.omega_param

    def _calculate_AGSK_pos(self):

        self.position_AGSK = np.zeros(3)

        self.position_AGSK[0] = self.position_norma_AGSK * (np.cos(self.u_param) * np.cos(self.Omega_param) - np.sin(self.u_param) * np.sin(self.Omega_param) * np.cos(self.i_param))
        self.position_AGSK[1] = self.position_norma_AGSK * (np.cos(self.u_param) * np.sin(self.Omega_param) + np.sin(self.u_param) * np.cos(self.Omega_param) * np.cos(self.i_param))
        self.position_AGSK[2] = self.position_norma_AGSK * np.sin(self.u_param) *  np.sin(self.i_param)
    
    def _calculate_vellocities(self):

        self.vel_vector = np.zeros(3)

        self.transversial_velocity = np.sqrt(self.earth_grav_param / self.Focal_param) * self.Ext_param * np.sin(self.theta_anomaly)
        self.radial_velocity = np.sqrt(self.earth_grav_param / self.Focal_param) * (1 + self.Ext_param * np.cos(self.theta_anomaly))
        self.abs_velocity = np.sqrt(self.transversial_velocity ** 2 + self.radial_velocity ** 2)

        self.vel_vector[0] = self.transversial_velocity
        self.vel_vector[1] = self.radial_velocity
        self.vel_vector[2] = 0.0
    
    def _calculate_GSK_pos(self, time):

        self.earth_angle = self.earth_anguler_vel * time
        self.position_GSK = np.zeros(3)
        
        self.position_GSK[0] = self.position_AGSK[0] * np.cos(self.earth_angle) + self.position_AGSK[1] * np.sin(self.earth_angle)
        self.position_GSK[1] = -self.position_AGSK[0] * np.sin(self.earth_angle) + self.position_AGSK[1] * np.cos(self.earth_angle)
        self.position_GSK[2] = self.position_AGSK[2]
    
    def _calculate_LBH_pos(self):

        self.position_LBH = np.zeros(3)
        self.D_param = np.sqrt(self.position_GSK[0] ** 2 + self.position_GSK[1] ** 2)
        
        if self.D_param == 0:

            self.B_param = (np.pi / 2) * self.position_GSK[2] / np.abs(self.position_GSK[2])
            self.L_param = 0
            self.H_param = self.position_GSK[2] * np.sin(self.B_param - self.SM_earth_axis * np.sqrt(1 - self.earth_ext_param * np.sin(self.B_param) ** 2))

        elif self.D_param > 0:

            self.La_param = np.arcsin(self.position_GSK[1] / self.D_param)
            
            if (self.position_GSK[1] < 0 and self.position_GSK[0] > 0):

                self.L_param = 2.0 * np.pi - self.La_param
            
            elif (self.position_GSK[1] < 0 and self.position_GSK[0] < 0):

                self.L_param = np.pi + self.La_param
            
            elif (self.position_GSK[1] > 0 and self.position_GSK[0] < 0):

                self.L_param = np.pi -self.La_param
            
            elif (self.position_GSK[1] > 0 and self.position_GSK[0] > 0):

                self.L_param = self.La_param
            
            
            if (self.position_GSK[2] == 0):

                self.B_param = 0
                self.H_param = self.D_param - self.SM_earth_axis
            
            else:

                position_norma_GSK = np.sqrt(self.position_GSK[0] ** 2 + self.position_GSK[1] ** 2 + self.position_GSK[2] ** 2)
                c_param = np.arcsin(self.position_GSK[2] / position_norma_GSK)
                p_param = (self.earth_ext_param * self.SM_earth_axis) / (2 * position_norma_GSK)

                s_past_param = 0
                b_param = c_param + s_past_param
                print((p_param * np.sin(2 * b_param)) // np.sqrt(1 - self.earth_ext_param * np.sin(b_param) ** 2))
                s_param = np.arcsin((p_param * np.sin(2 * b_param)) // np.sqrt(1 - self.earth_ext_param * np.sin(b_param) ** 2))


                while (s_param - s_past_param):

                    s_past_param = s_param
                    s_param = np.arcsin((p_param * np.sin(2 * b_param)) // np.sqrt(1 - self.earth_ext_param * np.sin(b_param) ** 2))
                
                self.B_param = b_param
                self.H_param = self.D_param * np.cos(self.B_param) + self.position_GSK[1] - self.SM_earth_axis * np.sqrt(1 - self.earth_ext_param * np.sin(self.B_param) ** 2)
            
    
    def _calculate_density(self):
        
        self.upper_500 = False

        if self.H_param < 500:
            
            print("[Test 1] !!!")
            self.upper_500 = False
            self.night_density_120 = np.zeros(6)
            for vector_number in range(self.night_density_coeffs_120.shape[1] - 1):

                self.night_density_120[vector_number] = self.constant_night_density * (self.night_density_coeffs_120[0, vector_number] + self.night_density_coeffs_120[1, vector_number] * self.H_param
                                                               + self.night_density_coeffs_120[2, vector_number] * self.H_param ** 2 
                                                               + self.night_density_coeffs_120[3, vector_number] * self.H_param ** 3
                                                                + self.night_density_coeffs_120[4, vector_number] * self.H_param ** 4
                                                                 + self.night_density_coeffs_120[5, vector_number] * self.H_param ** 5
                                                                  + self.night_density_coeffs_120[6, vector_number] * self.H_param ** 6) 
            print(f"density: {self.night_density_120}")
        elif (self.H_param > 500):
            
            print("[Test 2] !!!")
            self.upper_500 = True
            self.night_density_500 = np.zeros(6)
            for vector_number in range(self.night_density_coeffs_500.shape[1] - 1):

                self.night_density_500[vector_number] = self.constant_night_density * (self.night_density_coeffs_500[0, vector_number] + self.night_density_coeffs_500[1, vector_number] * self.H_param
                                                               + self.night_density_coeffs_500[2, vector_number] * self.H_param ** 2 
                                                               + self.night_density_coeffs_500[3, vector_number] * self.H_param ** 3
                                                                + self.night_density_coeffs_500[4, vector_number] * self.H_param ** 4
                                                                 + self.night_density_coeffs_500[5, vector_number] * self.H_param ** 5
                                                                  + self.night_density_coeffs_500[6, vector_number] * self.H_param ** 6) 


    def _calculate_acceleration(self):

        self.acceleration_tensor = np.zeros((6, 3))

        if self.upper_500 == False:
            
            
            for (density_number, density) in enumerate(self.night_density_120):
                
                acceleration_S = -density * self.abs_velocity * self.radial_velocity
                acceleration_T = -density * self.abs_velocity * self.transversial_velocity
                acceleration_ABS = np.sqrt(acceleration_T ** 2 + acceleration_S ** 2)

                print(f"density: {density}")
                print(f"density_number: {density_number}")
                print(f"abs vel: {self.abs_velocity}")
                print(f"radial vel: {self.radial_velocity}")
                print(f"transversial vel: {self.transversial_velocity}")

                self.acceleration_tensor[density_number, 0] = acceleration_S
                self.acceleration_tensor[density_number, 1] = acceleration_T
                self.acceleration_tensor[density_number, 2] = acceleration_ABS
        
        else:

            for (density_number, density) in enumerate(self.night_density_500):
                
                acceleration_S = -density * self.abs_velocity * self.radial_velocity
                acceleration_T = -density * self.abs_velocity * self.transversial_velocity
                acceleration_ABS = np.sqrt(acceleration_T ** 2 + acceleration_S ** 2)

                print(f"density: {density}")
                print(f"density_number: {density_number}")
                print(f"abs vel: {self.abs_velocity}")
                print(f"radial vel: {self.radial_velocity}")
                print(f"transversial vel: {self.transversial_velocity}")

                self.acceleration_tensor[density_number, 0] = acceleration_S
                self.acceleration_tensor[density_number, 1] = acceleration_T
                self.acceleration_tensor[density_number, 2] = acceleration_ABS
    

    def _show_data(self):

        plt.style.use("dark_background")
        self.fig, self.axis = plt.subplots(nrows=3)

        self.axis[0].plot(self.acceleration_tensor[:, 0], color="r", label="acceleration S")
        self.axis[1].plot(self.acceleration_tensor[:, 1], color="g", label="acceleration T")
        self.axis[2].plot(self.acceleration_tensor[:, 2], color="b", label="acceleration ABS")

        self.axis[0].legend(loc="upper left")
        self.axis[1].legend(loc="upper left")
        self.axis[2].legend(loc="upper left")

        self.axis[0].grid()
        self.axis[1].grid()
        self.axis[2].grid()
        plt.show()
    
    def _start_simulation(self, max_time):

        self.time = 1
        self._calculate_anomalies()
        self._calculate_AGSK_pos()
        self._calculate_vellocities()



        self._calculate_GSK_pos(time=self.time)
        self._calculate_LBH_pos()
        self._calculate_density()
        self._calculate_acceleration()

        print(self.acceleration_tensor)
        self._show_data()
    
if __name__ == "__main__":
    
    sim_object = MechanicCalculator()
    max_time = np.sqrt(sim_object.Sm_axis ** 3 / sim_object.earth_grav_param)
    sim_object._start_simulation(max_time=max_time)


            
