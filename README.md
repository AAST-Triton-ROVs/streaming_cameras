# streaming_cameras
codes for streaming camera in station 

# In rpi inside rov

1- In home directory enter old_vision directory : cd old_pi

2- activate .venv : source .venv/bin/activate

3-Enter vision directory : cd vision

4-run udp_sender : python3 udp_sender



# In station:

1- In home directory (triton_desktop) activate vision_venv : source vision_venv/bin/activate

2- Enter vision directory : cd vision   

3- Enter Color Filters directory : cd Color Filters                                                                                                          

4- run station_reciver file : python3 station_reciever\



# run the code in rpi first then in the station
