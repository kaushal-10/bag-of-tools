# Commands to get device specific details


### Read the temperature of the device
awk '{print $1/1000 "°C"}' /sys/class/thermal/thermal_zone*/temp