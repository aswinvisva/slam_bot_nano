# Give permission to access usb port
sudo chmod 666 /dev/ttyUSB0

# Restart nvargs service... idk why
sudo service nvargus-daemon restart

# Run the script
python3 record.py