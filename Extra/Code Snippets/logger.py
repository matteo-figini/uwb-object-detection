import json
import pyqtgraph as pg
import matplotlib.image as mpimg
import numpy as np
import threading
import os
import time
import serial
from scipy import signal, constants

from serial.tools import list_ports
import sounddevice as sd
from pyqtgraph.Qt.QtWidgets import (QGraphicsProxyWidget, QLineEdit, QPushButton, QLabel, 
                                    QFormLayout, QWidget, QVBoxLayout, QComboBox, QListView, 
                                    QRadioButton, QGraphicsEllipseItem, QButtonGroup, 
                                    QGraphicsRectItem, QMessageBox, QGroupBox, QHBoxLayout, QMainWindow, QCheckBox)
from pyqtgraph.Qt.QtCore import QRegExp, QSize, QThread, pyqtSignal, Qt, pyqtSlot
from pyqtgraph.Qt.QtGui import QRegExpValidator


class InfineonSignalProcessing(QThread):
    collection_finished = pyqtSignal(object, str)
    signalLive = pyqtSignal()
    def __init__(self, stop_event,fps):
        super().__init__()
        self.fps=fps
        self.stop_event = stop_event

        self.num_ant = 3
        self.num_chirps = 4
        self.samples_per_chirp = 128



        self.bytes_per_cir = self.samples_per_chirp * self.num_chirps *self.num_ant * 2

        self.frame = np.empty((0), dtype = np.uint8)
        self.append_flag = False




    def run(self):
        self.start_radar() #return when total_samples_required are collected
        if not self.stop_event.is_set():
            self.save_data()
        #self.stop_event.clear()
        print("DATA ACQUISITION FINISHED!")


    def set_parameters(self, device, samples_number, window_duration, user_id, activity, room, target_position, timestamp):
        self.samples_number = samples_number
        self.window_duration = window_duration
        self.user_id = user_id
        self.activity = activity
        self.room = room
        self.target_position = target_position
        self.timestamp = timestamp
        self.total_samples_required = self.samples_number * (self.fps * self.window_duration) + self.fps #add a second to have enough samples for decluttering

        self.frames  = np.zeros((self.total_samples_required, self.num_ant, self.num_chirps, self.samples_per_chirp), dtype=np.float32)
        self.samples_collected = 0
 
        print(f"Starting radar data acquisition for {self.user_id}...")
        print(f"Samples number: {self.samples_number}, Window duration: {self.window_duration} s")

        self.ser = serial.Serial(device)



    def start_radar(self):

        len_antenna = self.num_chirps*self.samples_per_chirp


        try:
            self.ser.write(b"START")

            while not self.stop_event.is_set():

                data = self.ser.readline()

                #print(data)

                if not self.append_flag:

                    if data == b"BEGIN\n":

                        self.append_flag=True
                
                else:

                    if(data == b"END\n"):
                        
                        self.frame = self.frame[:-1]

                        if(self.frame.shape[0]==self.bytes_per_cir):
                            self.frame = self.frame.view(np.int16)
                            
                            rx1 = self.frame[:len_antenna].reshape(self.num_chirps,self.samples_per_chirp)
                            rx2 = self.frame[len_antenna:len_antenna*2].reshape(self.num_chirps,self.samples_per_chirp)
                            rx3 = self.frame[len_antenna*2:].reshape(self.num_chirps,self.samples_per_chirp)



                            self.frames[self.samples_collected, 0] = rx1
                            self.frames[self.samples_collected, 1] = rx2
                            self.frames[self.samples_collected, 2] = rx3

                            self.signalLive.emit()
                            self.samples_collected +=1

                            if self.samples_collected == self.total_samples_required:
                                break

                        else:

                            print("Frame of shape ",self.frame.shape, "discarded")

                        self.frame = np.empty((0), dtype=np.uint8 )

                        self.append_flag = False


                        

                    else:

                        self.frame = np.concatenate([self.frame, np.frombuffer(data, dtype=np.uint8)])

            self.ser.write(b"STOP")
            print("INFINEON STOPPED")
            self.ser.close()
        
        except Exception as e:
            print(f"Error: {e}")

        





    def stop_acquisition(self):


        #self.save_data()

        self.stop_event.set()

        #self.ser.close()
        
        print(f"Stopping radar data acquisition for {self.user_id}...")

    

    def save_data(self):

        file_list = []
        filepath = "datasets/Infineon"

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        filename=f"{self.user_id}_{self.activity}"

        if self.room:
            filename += f"_{self.room}"

        if self.target_position:
            filename += f"_{self.target_position}"

        filename += f"_{self.timestamp}"

        filepath = os.path.join(filepath,filename)

        print(self.frames.shape)

        for i in range(self.num_ant):

            data=self.frames[:,i,:,:]
            file = filepath+f"_Infineon_rx{i}.npy"

            np.save(file, data)

            file_list.append(file)

        self.collection_finished.emit(file_list, "Infineon")



class SR250DevKitSignalProcessing(QThread):
    collection_finished = pyqtSignal(object,str)
    signalLive = pyqtSignal()

    def __init__(self, stop_event,fps):
        super().__init__()
        self.fps=fps
        self.stop_event = stop_event

        self.taps = 128
        
        self.range_bins = 120
        self.num_ant = 3
        self.bytes_per_cir = self.taps * 4 *self.num_ant

        self.frame = np.empty((0), dtype = np.uint8)
        self.append_flag = False




    def run(self):
        self.start_radar() #return when total_samples_required are collected
        if not self.stop_event.is_set():
            self.save_data()
        #self.stop_event.clear()
        print("DATA ACQUISITION FINISHED!")


    def set_parameters(self, device, samples_number, window_duration, user_id, activity, room, target_position, timestamp):
        self.samples_number = samples_number
        self.window_duration = window_duration
        self.user_id = user_id
        self.activity = activity
        self.room = room
        self.target_position = target_position
        self.timestamp = timestamp
        self.total_samples_required = self.samples_number * (self.fps * self.window_duration) + self.fps #add a second to have enough samples for decluttering

        self.frames  = np.zeros((self.total_samples_required, self.num_ant, self.range_bins), dtype=np.complex64)
        self.samples_collected = 0
 
        print(f"Starting radar data acquisition for {self.user_id}...")
        print(f"Samples number: {self.samples_number}, Window duration: {self.window_duration} s")

        self.ser = serial.Serial(device, 1500000, timeout=1)



    def start_radar(self):

        len_antenna = self.taps*2


        try:
            self.ser.write(b"START")

            while not self.stop_event.is_set():

                data = self.ser.readline()

                #print(data)

                if not self.append_flag:

                    if data == b"BEGIN\n":

                        self.append_flag=True
                
                else:

                    if(data == b"END\n"):
                        
                        self.frame = self.frame[:-1]

                        if(self.frame.shape[0]==self.bytes_per_cir):
                            self.frame = self.frame.view(np.int16)
                            
                            rx1 = self.frame[:len_antenna]
                            rx2 = self.frame[len_antenna:len_antenna*2]
                            rx3 = self.frame[len_antenna*2:] 
                            cir_casted_int16 = rx1[16:].reshape((len_antenna*2 - 32) // 4, 2)
                            rx1_complex = (cir_casted_int16[:, 0] + 1j * cir_casted_int16[:, 1]).astype(np.complex64)

                            cir_casted_int16 = rx2[16:].reshape((len_antenna*2 - 32) // 4, 2)
                            rx2_complex = (cir_casted_int16[:, 0] + 1j * cir_casted_int16[:, 1]).astype(np.complex64)

                            cir_casted_int16 = rx3[16:].reshape((len_antenna*2 - 32) // 4, 2)
                            rx3_complex = (cir_casted_int16[:, 0] + 1j * cir_casted_int16[:, 1]).astype(np.complex64)


                            self.frames[self.samples_collected, 0, :] = rx1_complex
                            self.frames[self.samples_collected, 1, :] = rx2_complex
                            self.frames[self.samples_collected, 2, :] = rx3_complex
                            self.signalLive.emit()
                            self.samples_collected +=1

                            if self.samples_collected == self.total_samples_required:
                                break

                        else:

                            print("Frame of shape ",self.frame.shape, "discarded")

                        self.frame = np.empty((0), dtype=np.uint8 )

                        self.append_flag = False


                        

                    else:

                        self.frame = np.concatenate([self.frame, np.frombuffer(data, dtype=np.uint8)])

            self.ser.write(b"STOP")
            print("SR250DEV STOPPED")
            self.ser.close()
        
        except Exception as e:
            print(f"Error: {e}")

        





    def stop_acquisition(self):


        #self.save_data()

        self.stop_event.set()

        #self.ser.close()
        
        print(f"Stopping radar data acquisition for {self.user_id}...")

    

    def save_data(self):

        file_list = []
        filepath = "datasets/SR250DevKit"

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        filename=f"{self.user_id}_{self.activity}"

        if self.room:
            filename += f"_{self.room}"

        if self.target_position:
            filename += f"_{self.target_position}"

        filename += f"_{self.timestamp}"
        

        filepath = os.path.join(filepath,filename)

        print(self.frames.shape)

        for i in range(self.num_ant):

            data=self.frames[:,i,:]
            file = f"{filepath}_sr250_rx{i}.npy"

            #data = data.view(np.float32)
            np.save(file, data)

            file_list.append(file)


        self.collection_finished.emit(file_list,"SR250DevKit")


class SR250MateSignalProcessing(QThread):
    collection_finished = pyqtSignal(object,str)
    signalLive = pyqtSignal()

    def __init__(self, stop_event,fps):
        super().__init__()
        self.fps=fps
        self.stop_event = stop_event

        self.taps = 128
        
        self.range_bins = 120
        self.num_ant = 3
        self.bytes_per_cir = self.taps * 4 *self.num_ant

        self.frame = np.empty((0), dtype = np.uint8)
        self.append_flag = False




    def run(self):
        self.start_radar() #return when total_samples_required are collected
        if not self.stop_event.is_set():
            self.save_data()
        #self.stop_event.clear()
        print("DATA ACQUISITION FINISHED!")


    def set_parameters(self, device, samples_number, window_duration, user_id, activity, room, target_position, timestamp):
        self.samples_number = samples_number
        self.window_duration = window_duration
        self.user_id = user_id
        self.activity = activity
        self.room = room
        self.target_position = target_position
        self.timestamp = timestamp
        self.total_samples_required = self.samples_number * (self.fps * self.window_duration) + self.fps #add a second to have enough samples for decluttering

        self.frames  = np.zeros((self.total_samples_required, self.num_ant, self.range_bins), dtype=np.complex64)
        self.samples_collected = 0
 
        print(f"Starting radar data acquisition for {self.user_id}...")
        print(f"Samples number: {self.samples_number}, Window duration: {self.window_duration} s")

        self.ser = serial.Serial(device, timeout=1)



    def start_radar(self):

        len_antenna = self.taps*2


        try:
            self.ser.write(b"START")

            while not self.stop_event.is_set():

                data = self.ser.readline()

                #print(data)

                if not self.append_flag:

                    if data == b"BEGIN\n":

                        self.append_flag=True
                
                else:

                    if(data == b"END\n"):
                        
                        self.frame = self.frame[:-1]

                        if(self.frame.shape[0]==self.bytes_per_cir):
                            self.frame = self.frame.view(np.int16)
                            
                            rx1 = self.frame[:len_antenna]
                            rx2 = self.frame[len_antenna:len_antenna*2]
                            rx3 = self.frame[len_antenna*2:] 
                            cir_casted_int16 = rx1[16:].reshape((len_antenna*2 - 32) // 4, 2)
                            rx1_complex = (cir_casted_int16[:, 0] + 1j * cir_casted_int16[:, 1]).astype(np.complex64)

                            cir_casted_int16 = rx2[16:].reshape((len_antenna*2 - 32) // 4, 2)
                            rx2_complex = (cir_casted_int16[:, 0] + 1j * cir_casted_int16[:, 1]).astype(np.complex64)

                            cir_casted_int16 = rx3[16:].reshape((len_antenna*2 - 32) // 4, 2)
                            rx3_complex = (cir_casted_int16[:, 0] + 1j * cir_casted_int16[:, 1]).astype(np.complex64)


                            self.frames[self.samples_collected, 0, :] = rx1_complex
                            self.frames[self.samples_collected, 1, :] = rx2_complex
                            self.frames[self.samples_collected, 2, :] = rx3_complex
                            self.signalLive.emit()
                            self.samples_collected +=1

                            if self.samples_collected == self.total_samples_required:
                                break

                        else:

                            print("Frame of shape ",self.frame.shape, "discarded")

                        self.frame = np.empty((0), dtype=np.uint8 )

                        self.append_flag = False


                        

                    else:

                        self.frame = np.concatenate([self.frame, np.frombuffer(data, dtype=np.uint8)])

            self.ser.write(b"STOP")
            self.ser.close()
        
        except Exception as e:
            print(f"Error: {e}")

        





    def stop_acquisition(self):


        #self.save_data()

        self.stop_event.set()

        #self.ser.close()
        
        print(f"Stopping radar data acquisition for {self.user_id}...")

    

    def save_data(self):

        file_list = []
        filepath = "datasets/SR250Mate"

        if not os.path.exists(filepath):
            os.mkdir(filepath)


        filename=f"{self.user_id}_{self.activity}"

        if self.room:
            filename += f"_{self.room}"

        if self.target_position:
            filename += f"_{self.target_position}"

        filename += f"_{self.timestamp}"

        filepath = os.path.join(filepath,filename)

        print(self.frames.shape)

        for i in range(self.num_ant):

            data=self.frames[:,i,:]
            file = f"{filepath}_sr250_rx{i}.npy"

            #data = data.view(np.float32)
            np.save(file, data)

            file_list.append(file)

            
        self.collection_finished.emit(file_list, "SR250Mate")





class X7SignalProcessing(QThread):
    collection_finished = pyqtSignal(object)

    def __init__(self, stop_event, fps):
        super(X7SignalProcessing, self).__init__()

        self.stop_event = stop_event
        self.fps = fps
        self.num_tx = 2
        self.num_rx = 2
        self.num_antennas = self.num_tx * self.num_rx

        self.taps_before = 16
        self.taps = 112
        self.frame = np.empty((0), dtype=np.uint8)
        self.append_flag =  False
        self.discard = False
        self.tx = 0

        self.samples_collected = 0
        self.counter = 0

        self.win_number = 1
        self.window_size = 15

        self.user_id = "GenericUser"
        self.activity = "generic"

        
        

    def run(self):
        
        #while not self.stop_event.is_set():
            #print("Processing signal")
        #    time.sleep(2)

        self.start_radar() #return when total_samples_required are collected
        self.stop_event.clear()
        self.save_data()
        print("Stopping signal processing")

    
    


    
    def set_parameters(self, device, samples_number, window_size, user_id, activity, room, target_position, timestamp):
        
        self.samples_number = samples_number
        self.window_size = window_size
        self.user_id = user_id
        self.activity = activity
        self.room = room
        self.target_position = target_position
        self.timestamp = timestamp
        self.total_samples_required = self.samples_number * (self.fps * self.window_size) + self.fps

        self.samples_collected = 0
        self.counter = 0
        self.tx = 0

        self.frames = np.zeros((self.total_samples_required, self.num_tx, self.num_rx, self.taps), dtype = np.complex64)

        print(f"Starting radar data acquisition for {self.user_id}...")
        print(f"Samples number: {self.samples_number}, Window duration: {self.window_size} s")

        self.collect = True
        self.ser = serial.Serial(device, 2000000, timeout=1)
        
    

    def start_radar(self):

        try:
            self.ser.write(b"START")

            while True and not self.stop_event.is_set():

                data = self.ser.readline()

                if not self.append_flag:

                    if b"BEGIN" in data:

                        self.append_flag = True

                else:

                    if(data == b"END\n"):
                        self.frame =  self.frame[:-1]

                        if(self.frame.shape[0] !=  1792):
                            print(f"Frame length: {self.frame.shape[0]}")
                            print("Frame discarded")
                            self.frame = np.empty((0), dtype=np.uint8)
                            self.append_flag = False
                            self.discard = True

                        else:
                            self.frame = self.frame.view(np.float32)

                            rx1 = self.frame[:224]
                            rx2 = self.frame[224:]

                            rx1_complex = rx1[:112] + 1j * rx1[112:]
                            rx2_complex = rx2[:112] + 1j * rx2[112:]

                            self.frames[self.samples_collected, self.tx, 0, :] = rx1_complex 
                            self.frames[self.samples_collected, self.tx, 1, :] = rx2_complex

                            self.frame = np.empty((0), dtype=np.uint8)

                            self.append_flag = False


                        if(self.tx == 1):

                            if not self.discard:

                                self.samples_collected += 1

                                if self.samples_collected >= self.total_samples_required:
                                    print("COLLECTION FINISHED")
                                    self.ser.close()
                                    return
                            else:
                                self.discard = False
                            
                        self.tx = self.tx ^1

                    else:
                        self.frame = np.concatenate([self.frame, np.frombuffer(data, dtype = np.uint8)])
        
        except Exception as e:
            print(f"Error: {e}")


    def decluttering (self, cirs, alpha):

        result = []

        dec_base = cirs[0]*0

        normalization = (1+alpha)/2

        for i in range(cirs.shape[0]):

            result.append((cirs[i]- dec_base)*normalization)

            dec_base = dec_base * alpha + cirs[i] * (1-alpha)

        return result


    def save_data(self):
        
        file_list = []

        filepath = "dataset"

        if not os.path.exists(filepath):
            os.mkdir(filepath)

        filename = f"{self.user_id}_{self.activity}_{self.room}_{self.target_position}_{self.timestamp}"
        
        filepath = os.path.join(filepath, filename)

        for i in range(self.num_tx):
            for j in range(self.num_rx):

                self.frames[:, i, j, :] = self.decluttering(self.frames[:, i, j, :], 0.4)

                np.save(f"{filepath}_x7_tx{i}rx{j}.npy", self.frames[10:self.frames.shape[0]-15, i, j, 16:86])

                file_list.append(f"{filepath}_x7_tx{i}rx{j}.npy")

        self.collection_finished.emit(file_list)

        
class FormLayout(QWidget):
    def __init__(self):
        super().__init__()

        layout = QFormLayout()

        self.user_textbox = QLineEdit(placeholderText = "UserID")

        layout.addRow(QLabel("UserID: "), self.user_textbox)

        self.room_textbox = QLineEdit(placeholderText = "Stanza")

        self.room_textbox.textChanged.connect(self.add_special_activities)

        layout.addRow(QLabel("Stanza: "), self.room_textbox)

        self.activity_combobox = QComboBox(placeholderText = "Attività")

        listView = QListView(self.activity_combobox)

        with open("logger_conf.json", 'r') as file:
            conf = json.load(file)

        for activity in conf["activities"]:

            self.activity_combobox.addItem(activity)

        self.activity_combobox.setView(listView)
        layout.addRow(QLabel("Attività: "), self.activity_combobox)

        

        self.special_combobox = QComboBox(placeholderText = "Casi Speciali")
        special_listView = QListView(self.special_combobox)

        self.special_combobox.setView(special_listView)
        self.special_combobox.setVisible(False)
        self.special_label = QLabel("Casi Speciali: ")
        self.special_label.setVisible(False)
        layout.addRow(self.special_label, self.special_combobox)

        self.activity_combobox.currentTextChanged.connect(self.toggle_special_combobox)

        regex = QRegExp("[1-9]\\d*")

        self.window_size_textbox = QLineEdit(placeholderText="Dimensione finestra (s)")
        self.window_size_textbox.setText("15")
        self.window_size_textbox.setValidator(QRegExpValidator(regex))

        layout.addRow(QLabel("Dimensione finestra (s): "), self.window_size_textbox)

        self.samples_number_textbox = QLineEdit(placeholderText = "Numero finestre")
        self.samples_number_textbox.setText("1")
        self.samples_number_textbox.setValidator(QRegExpValidator(regex))

        layout.addRow(QLabel("Numero finestre: "), self.samples_number_textbox)

        self.setLayout(layout)

        self.timerComboBox = QComboBox(placeholderText = "Timer")
        timerListView = QListView(self.timerComboBox)
        self.timerComboBox.setView(timerListView)        
        self.timerComboBox.addItems([f"{i}" for i in range(11)])
        self.timerComboBox.setCurrentIndex(0)
        self.timerLabel = QLabel("Timer: ")
        layout.addRow(self.timerLabel, self.timerComboBox) 

        checkbox_layout = QHBoxLayout()
        #checkbox_layout.setAlignment(Qt.AlignRight)
        self.sr250active = QCheckBox("SR250(Mate)")
        self.sr250active.setLayoutDirection(Qt.RightToLeft)
        self.sr250active.setStyleSheet("QCheckBox { color: crimson; }")
        self.sr250active.toggled.connect(self.init_serial_sr250) 
        self.sr250DevActive = QCheckBox("SR250(DevKit)")
        self.sr250DevActive.setLayoutDirection(Qt.RightToLeft)
        self.sr250DevActive.setStyleSheet("QCheckBox { color: crimson; }") 
        self.sr250DevActive.toggled.connect(self.init_serial_sr250_dev)
        self.infineonActive = QCheckBox("Infineon")
        self.infineonActive.setLayoutDirection(Qt.RightToLeft)
        self.infineonActive.setStyleSheet("QCheckBox { color: crimson; }") 
        self.infineonActive.toggled.connect(self.init_serial_infineon)

        
        checkbox_layout.addWidget(self.sr250active)
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.sr250DevActive)
        checkbox_layout.addStretch()
        checkbox_layout.addWidget(self.infineonActive)

        #checkbox_layout.setAlignment(Qt.AlignCenter)

        layout.addRow(QLabel(""), checkbox_layout)


    def add_special_activities(self, text):
        
        if text == "Soggiorno":

            self.special_combobox.clear()
            self.special_combobox.addItem("Disteso sul divano")

        elif text == "Camera":

            self.special_combobox.clear()
            self.special_combobox.addItem("Disteso sul letto")
        
        elif text == "Bagno":

            self.special_combobox.clear()
            self.special_combobox.addItem("Disteso in vasca")

        else:
            self.special_combobox.clear()



    def toggle_special_combobox(self, text):

        if text == "Casi Speciali":
            self.special_combobox.setVisible(True)
            self.special_label.setVisible(True)
            if self.room_textbox.text() == "":
                
                QMessageBox.warning(self, "Error", "Inserisci la stanza per visualizzare i casi speciali")
            
        else:
            self.special_combobox.setVisible(False)
            self.special_label.setVisible(False)

    


    
    def init_serial_sr250(self):

        if self.sr250active.isChecked():
            ports = list_ports.comports()
            esp_port = None 

            for port in ports:
                try: 
                    ser = serial.Serial(port.device, timeout = 1)
                    ser.write(b'INFO\r\n')

                    response = ser.read(100).decode('utf-8', errors='ignore')

                    if "SR250" in response:
                        esp_port = port.device
                        ser.close()
                        print(f"Found SR250 device on port {esp_port}")
                        break
                except Exception as e:
                    print(f"Error checking port {port.device}: {e}")
                    continue

            if esp_port is None:
                print("ESP device not found. Exiting.")
                self.sr250active.setChecked(False)
            else:
                self.sr250active.setStyleSheet("QCheckBox { color: green; }") 

            self.sr250Port = esp_port

        else:
            self.sr250active.setStyleSheet("QCheckBox { color: crimson; }")
            self.sr250Port = None


    def init_serial_sr250_dev(self):

        if self.sr250DevActive.isChecked():
            ports = list_ports.comports()
            esp_port = None 

            for port in ports:
                try: 
                    ser = serial.Serial(port.device, baudrate = 1500000, timeout = 1)
                    ser.write(b'INFO\r\n')

                    response = ser.read(100).decode('utf-8', errors='ignore')

                    if "SR250" in response:
                        esp_port = port.device
                        ser.close()
                        print(f"Found SR250 device on port {esp_port}")
                        break
                except Exception as e:
                    print(f"Error checking port {port.device}: {e}")
                    continue

            if esp_port is None:
                print("ESP device not found. Exiting.")
                self.sr250DevActive.setChecked(False)
            else:
                self.sr250DevActive.setStyleSheet("QCheckBox { color: green; }") 

            self.sr250DevPort = esp_port

        else:
            self.sr250DevActive.setStyleSheet("QCheckBox { color: crimson; }") 
            self.sr250DevPort = None

    def init_serial_infineon(self):

        if self.infineonActive.isChecked():
            ports = list_ports.comports()
            
            esp_port = None 

            for port in ports:
                try: 
                    ser = serial.Serial(port.device, timeout = 1)
                    ser.write(b'INFO\r\n')

                    response = ser.read(100).decode('utf-8', errors='ignore')

                    if "Infineon" in response:
                        esp_port = port.device
                        ser.close()
                        print(f"Found Infineon device on port {esp_port}")
                        break
                    else:
                        print(f"Found device on port {port.device} but it is not an Infineon device")
                except Exception as e:
                    print(f"Error checking port {port.device}: {e}")
                    continue

            if esp_port is None:
                print("ESP device not found. Exiting.")
                self.infineonActive.setChecked(False)
            else:
                self.infineonActive.setStyleSheet("QCheckBox { color: green; }") 

            self.infineonPort = esp_port

        else:
            self.infineonPort = None
            self.infineonActive.setStyleSheet("QCheckBox { color: crimson; }") 



class Logger(pg.GraphicsView):
    def __init__(self):
        super(Logger, self).__init__()

        self.visualizationMode = "Heatmap"

        #self.grid_items = []  # To track polar grid elements

        

        self.init_logger()
        l = pg.GraphicsLayout()
        self.setCentralItem(l)
        self.setBackground('w')
        self.setWindowTitle("TRUESENSE - Fall Detection Dataset Collector")
        self.showMaximized()

        logo = mpimg.imread('./assets/logo.png')
        logo = logo.transpose((1, 0, 2))
        logo = np.flip(logo, 1)

        logoitem = pg.ImageItem(logo)
        logoitem.setImage(logo)

        logobox = l.addViewBox(border='w', colspan=2)
        logobox.setAspectLocked()
        logobox.addItem(logoitem)
        logobox.setFixedHeight(80)

        l.nextRow()

        title = l.addLabel("UWB Dataset Collector",  border='w', size='24pt', bold=True, color='black', anchor=(0.5,0), colspan=2)

        l.nextRow()

        self.form = FormLayout()

        container = QWidget()
        v_layout = QVBoxLayout(container)
        v_layout.addWidget(self.form)

        container.setLayout(v_layout)

        container.setObjectName("formContainer")

        container.setStyleSheet("QWidget#formContainer {background-color: white; padding: 10px;}")

        container_proxy = QGraphicsProxyWidget()
        container_proxy.setWidget(container)

        l.addItem(container_proxy, colspan=1)

        self.plot_item = pg.PlotItem()

        self.plot_item.setAspectLocked(True)
        self.plot_item.getViewBox().setRange(xRange=(-15, 500), yRange=(-40, 500), padding=0)
        self.plot_item.getViewBox().setMouseEnabled(x=False, y=False)
        #self.plot_item.setMinimumSize(400,400)
        #self.plot_item.setFixedWidth(700)
        l.layout.setColumnStretchFactor(0, 1)
        self.create_polar_grid()
        self.create_radio_buttons()
        self.plot_item.hide()

        l.addItem(self.plot_item, colspan=1, rowspan=2)


        
        
        self.button = QPushButton("Change Visualization")
        self.button.setFixedWidth(300)

        self.change_button_proxy = self.make_centered_proxy(self.button)
        
        l.addItem(self.change_button_proxy, colspan=1, row=3, col=0)
        self.button.clicked.connect(self.change_visualization)  # Connect button to function")

        self.img = []
        self.plt = []
        
        self.range_bins = 120
        self.alpha = 0.9
        self.normalization = (1+self.alpha)/2
        self.decBase = np.empty((3, self.range_bins), dtype = np.complex64)
        imgdata = np.zeros((self.range_bins, int(20*15)), dtype=np.float32)

        self.img.append(pg.ImageItem(border="w"))
        self.img[0].setImage(imgdata)
        self.img[0].setColorMap("viridis")
        self.plt.append(l.addPlot(anchor=(1,0), colspan=1,col=1, rowspan=1, row=2))
        
        self.plt[0].addItem(self.img[0])
        self.plt[0].setXRange(0,self.range_bins)
        self.plt[0].setTitle(f"SR250", size="30pt", bold=True, color="black")
        self.plt[0].getViewBox().autoRange()

        #l.nextRow()
        #l.addItem(self.sr250_plot, col=1, row=2)
        l.nextRow()
        

        self.img.append(pg.ImageItem(border="w"))
        self.img[1].setImage(imgdata)
        self.img[1].setColorMap("viridis")
        self.plt.append(l.addPlot(anchor=(1,0), colspan=1 ,col=1, rowspan=1, row =3))
        self.plt[1].addItem(self.img[1])
        self.plt[1].setXRange(0,self.range_bins)
        self.plt[1].setTitle(f"Infineon", size="30pt", bold=True, color="black")
        #l.addItem(self.infineon_plot, col=1, row =3)
        self.plt[1].getViewBox().autoRange()

        l.nextRow()

        self.start_btn_proxy = QGraphicsProxyWidget()
        self.start_button = QPushButton("Start Collection")
        self.start_button.clicked.connect(self.start_collection)
        self.start_btn_proxy.setWidget(self.start_button)

        l.addItem(self.start_btn_proxy, colspan=1)

        self.stop_btn_proxy = QGraphicsProxyWidget()
        self.stop_button = QPushButton("Stop Collection")
        self.stop_button.clicked.connect(self.stop_collection)
        self.stop_btn_proxy.setWidget(self.stop_button)

        l.addItem(self.stop_btn_proxy, colspan=1)


        self.show()
        
    def make_centered_proxy(self,widget):
        container = QWidget()
        container.setObjectName("buttonContainer")
        container.setStyleSheet("QWidget#buttonContainer {background-color: white; padding: 10px;}")
        layout = QVBoxLayout(container)
        layout.setAlignment(Qt.AlignHCenter)
        layout.addWidget(widget)
        container.setLayout(layout)

        proxy = QGraphicsProxyWidget()
        proxy.setWidget(container)
        return proxy

        


    def init_logger(self):

        self.stop_event = threading.Event()

        """while True:

            self.device = self.init_serial_sr250()

            if(self.device):
                break

            time.sleep(10)"""

        with open('logger_conf.json', 'r') as f:
            self.config = json.load(f)

            self.fps = self.config["fps"]        



    def create_polar_grid(self):
        """ Draws the radar plot grid """

        self.distance = [150, 250, 350, 450] 
        self.angles = [0, 22.5, 45, 67.5, 90]  

        self.range_ticks=[[(self.distance[i],f"{str(self.distance[i]/100)}m") for i in range(len(self.distance))]]
        
        self.plot_item.getAxis('bottom').setTicks(self.range_ticks)
        self.plot_item.getAxis('left').setTicks(self.range_ticks)
        self.plot_item.getAxis('bottom').setTextPen('b')
        self.plot_item.getAxis('left').setTextPen('b')
        
        for r in self.distance:
            # Draw circles for each radial distance
            circle = QGraphicsEllipseItem(-r, -r, 2*r, 2*r)
            circle.setPen(pg.mkPen('k', width=1))
            circle.setStartAngle(0)
            circle.setSpanAngle(-90*16)
            self.plot_item.addItem(circle)
            #self.grid_items.append(circle)
        
        # Draw angular lines
        for angle in self.angles:
            angle_rad = np.radians(angle)
            line = pg.PlotDataItem([0, np.cos(angle_rad)*max(self.distance)],
                                   [0, np.sin(angle_rad)*max(self.distance)],
                                   pen=pg.mkPen('k', width=1))
            targetItem2 = pg.TargetItem(
                pos=(np.cos(angle_rad)*max(self.distance), np.sin(angle_rad)*max(self.distance)),
                size=1,
                symbol="x",
                pen=pg.mkPen(None),
                label=f"{angle}°",
                labelOpts={
                    "offset": pg.QtCore.QPoint(15, 15),
                    "color": "#558B2F"
                }
            )
            self.plot_item.addItem(line)
            #self.grid_items.append(line)
            self.plot_item.addItem(targetItem2)
            #self.grid_items.append(targetItem2)

            rect = QGraphicsRectItem(pg.QtCore.QRectF(-20, -30, 40, 60))  # (x, y, width, height)
            rect.setPen(pg.QtGui.QPen(pg.mkColor('r'), 2))
            rect.setTransformOriginPoint(rect.rect().center())
            rect.setRotation(45)
            

            label = pg.QtWidgets.QGraphicsTextItem("RADAR")
            label.setPos(-10, 25)
            label.setRotation(45)  
            label.setTransform(pg.QtGui.QTransform().fromScale(1,-1))

        # Set the text color of the label
            label.setDefaultTextColor(pg.QtGui.QColor('red'))  # Set text color to blue

            font = pg.QtGui.QFont("Arial", 12, pg.QtGui.QFont.Bold)
            label.setFont(font)
            self.plot_item.addItem(label)
            #self.grid_items.append(label)


            self.plot_item.addItem(rect)
            #self.grid_items.append(rect)

    def change_visualization(self):
        
        if self.visualizationMode == "Heatmap": 
            self.plot_item.show()
            self.plt[0].hide()
            self.plt[1].hide()
            self.visualizationMode="Grid"
        else:
            self.plot_item.hide()
            self.plt[0].show()
            self.plt[1].show()
            self.visualizationMode="Heatmap"

    def show_polar_grid(self):
        self.plot_item.show()
        self.visualizationMode="Grid"


    def create_radio_buttons(self):
        """ Creates radio buttons and places them at appropriate grid points """
        self.positions = [
            (150, 0), (150, 22.5), (150, 45), (150, 67.5),(150, 90),  # Row 1
            (250, 0), (250, 22.5), (250, 45), (250, 67.5), (250, 90), # Row 2
            (350, 0), (350, 22.5), (350, 45), (350, 67.5), (350, 90), # Row 3
            (450, 0), (450, 22.5), (450, 45), (450, 67.5), (450, 90),   # Row 4
        ]
        self.button_group = QButtonGroup(self)

        self.last_checked = None


        # Create radio buttons for each position
        for idx, (r, angle_deg) in enumerate(self.positions, start=1):
            button = QRadioButton(str(idx))
            #button.setFixedHeight(1)
            button.setAutoExclusive(True)  #only 1 button to be checked at once
            
            self.button_group.addButton(button)
            
            # Calculate polar to Cartesian conversion for positioning the buttons
            angle_rad = np.radians(angle_deg)
            x = r * np.cos(angle_rad)-5
            y = r * np.sin(angle_rad)+12            
            
            # Map data coordinates to screen (scene) coordinates
            button_proxy = pg.QtWidgets.QGraphicsProxyWidget()
            button_proxy.setWidget(button)

            button_proxy.setTransformOriginPoint(button_proxy.rect().center())

            button_proxy.setTransform(pg.QtGui.QTransform().fromScale(1,-1))


            button_proxy.setPos(x,y)
        
            self.plot_item.addItem(button_proxy)
            #self.grid_items.append(button_proxy)
        self.button_group.buttonClicked.connect(self.toggle_radio)

    def toggle_radio(self,radio):
        
        if self.last_checked == radio:
            # Same button clicked again — toggle it off
            self.button_group.setExclusive(False)
            radio.setChecked(False)
            self.button_group.setExclusive(True)
            self.last_checked = None
        else:
            # New selection
            self.last_checked = radio
        

        
    
    def start_collection(self):

        print("Start Collection")
        self.stop_event.clear()

        if (self.form.sr250active.isChecked() or self.form.sr250DevActive.isChecked() or self.form.infineonActive.isChecked()):
            
            self.username = self.form.user_textbox.text()

            if(self.form.activity_combobox.currentText()== ""):
                QMessageBox.warning(self, "Error", "Seleziona l'attività")
                return
            else:

                self.activity = self.form.activity_combobox.currentText()

                if(self.activity == "Casi Speciali"):
                    if(self.form.special_combobox.currentText()==""):
                        QMessageBox.warning(self, "Error", "Seleziona il caso speciale!")
                        return
                    

                    else:

                        self.special_case = self.form.special_combobox.currentText()

                        self.activity += "_" + self.special_case

            self.room = self.form.room_textbox.text()

            self.samples_number = int(self.form.samples_number_textbox.text())

            self.window_duration = int(self.form.window_size_textbox.text())

            timestamp = time.strftime("%Y%m%d-%H%M%S")

            if self.visualizationMode == "Grid":
                if(self.button_group.checkedButton() is None):
                    QMessageBox.warning(self, "Error", "Seleziona la posizione del target")
                    return
                else:
                    self.selected_pos = str(self.positions[int(self.button_group.checkedButton().text())-1]).replace(" ","")
            else:
                if(self.button_group.checkedButton() is None):
                    self.selected_pos= None
                else:
                    self.selected_pos = str(self.positions[int(self.button_group.checkedButton().text())-1]).replace(" ","")

            if self.username == '':
                self.username = "GenericUser"

            time.sleep(int(self.form.timerComboBox.currentText()))
            t = np.linspace(0, 0.5, int(44100 * 0.5), endpoint=False)
            wave = 0.5 * np.sin(2 * np.pi * 440 * t)

            # Play the generated sound
            sd.play(wave, 44100)
            sd.wait()  # Wait until the sound is finished

            self.firstDec = [True, True, True]

            if self.form.sr250active.isChecked():
                self.sr250_radar = SR250MateSignalProcessing(stop_event=self.stop_event, fps = self.fps)
                self.sr250_radar.collection_finished.connect(self.save_message)
                self.sr250_radar.signalLive.connect(self.show_250_hmap)
                self.sr250_radar.set_parameters(self.form.sr250Port, self.samples_number, self.window_duration, self.username, self.activity, self.room, self.selected_pos, timestamp)
                self.dec_frames_sr250 = np.zeros((self.sr250_radar.total_samples_required,  self.sr250_radar.range_bins), dtype=np.complex64)
                self.sr250_samples_collected = 0
            if self.form.sr250DevActive.isChecked():
                self.sr250dev_radar = SR250DevKitSignalProcessing(stop_event=self.stop_event, fps = self.fps)
                self.sr250dev_radar.collection_finished.connect(self.save_message)
                self.sr250dev_radar.signalLive.connect(self.show_250_dev_hmap)
                self.sr250dev_radar.set_parameters(self.form.sr250DevPort, self.samples_number, self.window_duration, self.username, self.activity, self.room, self.selected_pos, timestamp)
                self.dec_frames_sr250dev = np.zeros((self.sr250dev_radar.total_samples_required,  self.sr250dev_radar.range_bins), dtype=np.complex64)
                self.sr250dev_samples_collected = 0
            if self.form.infineonActive.isChecked():
                self.infineon_radar = InfineonSignalProcessing(stop_event=self.stop_event, fps = self.fps)
                self.infineon_radar.collection_finished.connect(self.save_message)
                self.infineon_radar.signalLive.connect(self.show_infineon_hmap)
                self.infineon_radar.set_parameters(self.form.infineonPort, self.samples_number, self.window_duration, self.username, self.activity, self.room, self.selected_pos, timestamp)
                self.dec_frames_infineon = np.zeros((self.infineon_radar.total_samples_required,  self.infineon_radar.samples_per_chirp - 8), dtype=np.complex64)
                self.infineon_samples_collected = 0

            if self.form.sr250active.isChecked():
                self.sr250_radar.start()
            if self.form.sr250DevActive.isChecked():
                self.sr250dev_radar.start()
            if self.form.infineonActive.isChecked():
                self.infineon_radar.start()


        else:

            QMessageBox.warning(self, "Error", "Connetti almeno un device!")
            return
                





    
    def save_message(self, file_list, device_name):

        msg_box =  QMessageBox()

        msg_box.setWindowTitle("Conferma salvataggio")
        msg_box.setText(f"Raccolta dati per {device_name} completata! Vuoi salvarla?")
        msg_box.setIcon(QMessageBox.Question)

        yes_btn = msg_box.addButton(QMessageBox.Save)
        no_btn = msg_box.addButton(QMessageBox.Cancel)

        msg_box.exec_()

        if(msg_box.clickedButton() == yes_btn):

            print("SAVED")

        elif(msg_box.clickedButton() == no_btn):

            for f in file_list:
                os.remove(f)
            print("DISCARDED")



    def stop_collection(self):

        self.stop_event.set()

        print("Stop Collection")



    def decluttering(self, cir, rx):

        cir_abs = cir

        if self.firstDec[rx]:

            self.decBase[rx] = cir_abs

            self.firstDec[rx] = False
            return cir_abs*0
        
        else:

            res = ((self.alpha * self.decBase[rx]) + (1-self.alpha)*cir_abs)
            self.decBase[rx] = res

            res = (cir_abs - res)

            return res
        
    def decluttering_alt (self, cir, rx):
        cir_abs = cir

        if self.firstDec[rx]:

            self.decBase[rx] = cir_abs

            self.firstDec[rx] = False
            return cir_abs*0
        
        else:

            res = (cir - self.decBase[rx]) * self.normalization

            self.decBase[rx] = self.decBase[rx] * self.alpha + cir * (1-self.alpha)

            return res

        
    def fft_spectrum(self, mat, range_window):
        # Calculate fft spectrum
        # mat:          chirp data
        # range_window: window applied on input data before fft

        # received data 'mat' is in matrix form for a single receive antenna
        # each row contains 'num_samples' for a single chirp
        # total number of rows = 'num_chirps'

        # -------------------------------------------------
        # Step 1 - remove DC bias from samples
        # -------------------------------------------------
        [num_chirps, num_samples] = np.shape(mat)

        # helpful in zero padding for high resolution FFT.
        # compute row (chirp) averages
        avgs = np.average(mat, 1).reshape(num_chirps, 1)

        # de-bias values
        mat = mat - avgs
        # -------------------------------------------------
        # Step 2 - Windowing the Data
        # -------------------------------------------------
        mat = np.multiply(mat, range_window)

        # -------------------------------------------------
        # Step 3 - add zero padding here
        # -------------------------------------------------
        zp1 = np.pad(mat, ((0, 0), (0, num_samples)), 'constant')

        # -------------------------------------------------
        # Step 4 - Compute FFT for distance information
        # -------------------------------------------------
        range_fft = np.fft.fft(zp1) / num_samples

        # ignore the redundant info in negative spectrum
        # compensate energy by doubling magnitude
        range_fft = 2 * range_fft[:, range(int(num_samples))]

        return range_fft
    
    @pyqtSlot()
    def show_250_hmap(self):

        self.dec_frames_sr250[self.sr250_samples_collected,:] = self.decluttering_alt(self.sr250_radar.frames[self.sr250_samples_collected,0,:], 0)
        self.sr250_samples_collected += 1
        self.img[0].setImage(np.abs(self.dec_frames_sr250).T, autolevels = True)
        self.plt[0].getViewBox().autoRange()

    @pyqtSlot()
    def show_250_dev_hmap(self):

        self.dec_frames_sr250dev[self.sr250dev_samples_collected,:] = self.decluttering_alt(self.sr250dev_radar.frames[self.sr250dev_samples_collected,0,:], 1)
        self.sr250dev_samples_collected += 1
        self.img[0].setImage(np.abs(self.dec_frames_sr250dev).T, autolevels = True)
        self.plt[0].getViewBox().autoRange()



    @pyqtSlot()
    def show_infineon_hmap(self):

        self.range_window = signal.windows.blackmanharris(128).reshape(1, 128)
        
        data = 2 * self.infineon_radar.frames[self.infineon_samples_collected,0,:] / 4095 -1.0 
        data = self.fft_spectrum(data, self.range_window)
        data = np.divide(data.sum(axis=0), 4)
        self.dec_frames_infineon[self.infineon_samples_collected,:] = self.decluttering_alt(data[:120],2)
        
        self.infineon_samples_collected += 1

        self.img[1].setImage(np.abs(self.dec_frames_infineon).T, autolevels = True)
        self.plt[1].getViewBox().autoRange()



if __name__ == '__main__':

    app = pg.mkQApp("TRUESENSE - Fall Detection Dataset Collector")

    with open('./UbuntuStyle.css', 'r') as f:
        app.setStyleSheet(f.read())

    view = Logger()

        # Force fullscreen workaround
    main_window = QMainWindow()
    main_window.setCentralWidget(view)
    main_window.setWindowTitle(view.windowTitle())

    screen_geometry = app.primaryScreen().availableGeometry()
    main_window.setGeometry(screen_geometry)        # Force resize to screen size
    main_window.move(screen_geometry.topLeft())     # Move to top-left corner
    main_window.show()                              # Important: show before setting state
    main_window.setWindowState(Qt.WindowMaximized)  # Explicitly set window state

    pg.exec()
