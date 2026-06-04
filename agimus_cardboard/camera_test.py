from agimus_cardboard.CameraBasler import CameraBasler

cam = CameraBasler(serial_number="22312466",
                   exposure_us=60000,
                   packet_size=1500,
                   gain=0)

img = cam.capture_image()

print(img.shape)