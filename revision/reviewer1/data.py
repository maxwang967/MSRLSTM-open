import numpy as np


class MSData:
    def __init__(self, train_path, validate_path):
        self.train_data = None
        self.train_x = None
        self.train_y = None
        if train_path is not None:
            self.train_data = np.load(train_path)
        self.validate_data = np.load(validate_path)
        if train_path is not None:
            self.train_x = self.train_data["x"]
        if train_path is not None:
            self.train_y = self.train_data["y"]
        self.validate_x = self.validate_data["x"]
        self.validate_y = self.validate_data["y"]
        
    def get_lacc_x_data(self):
        return None if self.train_data is None else self.train_x[:, :, 9:10], self.validate_x[:, :, 9:10]
    
    def get_lacc_y_data(self):
        return None if self.train_data is None else self.train_x[:, :, 10:11], self.validate_x[:, :, 10:11]

    def get_lacc_z_data(self):
        return None if self.train_data is None else self.train_x[:, :, 11:12], self.validate_x[:, :, 11:12]

    def get_gyr_x_data(self):
        return None if self.train_data is None else self.train_x[:, :, 6:7], self.validate_x[:, :, 6:7]
    
    def get_gyr_y_data(self):
        return None if self.train_data is None else self.train_x[:, :, 7:8], self.validate_x[:, :, 7:8]

    def get_gyr_z_data(self):
        return None if self.train_data is None else self.train_x[:, :, 8:9], self.validate_x[:, :, 8:9]

    def get_mag_x_data(self):
        return None if self.train_data is None else self.train_x[:, :, 12:13], self.validate_x[:, :, 12:13]
    
    def get_mag_y_data(self):
        return None if self.train_data is None else self.train_x[:, :, 13:14], self.validate_x[:, :, 13:14]

    def get_mag_z_data(self):
        return None if self.train_data is None else self.train_x[:, :, 14:15], self.validate_x[:, :, 14:15]

    def get_pressure_data(self):
        return None if self.train_data is None else self.train_x[:, :, 19:20], self.validate_x[:, :, 19:20]
    
    def get_train_y_data(self):
        return None if self.train_data is None else self.train_y

    def get_validate_y_data(self):
        return self.validate_y
