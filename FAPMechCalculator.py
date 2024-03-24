import numpy as np
import matplotlib.pyplot as plt
import os



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
            
            elif (self.position_GSk[1] < 0 and self.position_GSK[0] < 0):

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
                s_param = np.arsin((self.Focal_param * np.sin(2 * b_param)) / np.sqrt(1 - self.earth_ext_param * np.sin(b_param) ** 2))


                while (s_param - s_past_param):

                    s_past_param = s_param
                    s_param = np.arsin((self.Focal_param * np.sin(2 * b_param)) / np.sqrt(1 - self.earth_ext_param * np.sin(b_param) ** 2))
                
                self.B_param = b_param
                self.H_param = self.D_param * np.cos(self.B_param) + self.position_GSK[1] - self.SM_earth_axis * np.sqrt(1 - self.earth_ext_param * np.sin(self.B_param) ** 2)
            
    
    # TODO дописать функцию вычисления плотности
    # TODO дописать функицю вычисления ускорений

            
