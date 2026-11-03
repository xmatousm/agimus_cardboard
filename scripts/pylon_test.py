from pypylon import pylon, genicam

def get_node(nodemap, name):
    try:
        node = nodemap.GetNode(name)
        if node is None:
            return None
        if not genicam.IsAvailable(node):
            return None
        return node
    except genicam.LogicalErrorException:
        return None


camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

try:
    camera.Open()
    nodemap = camera.GetNodeMap()

    # Conservative GigE packet size
    node = get_node(nodemap, "GevSCPSPacketSize")
    if node and genicam.IsWritable(node):
        node.SetValue(1500)
        print("Set GevSCPSPacketSize = 1500")

    # Some cameras use AcquisitionFrameRateEnable + AcquisitionFrameRate
    node = get_node(nodemap, "AcquisitionFrameRateEnable")
    if node and genicam.IsWritable(node):
        node.SetValue(True)
        print("Enabled AcquisitionFrameRate")

    node = get_node(nodemap, "AcquisitionFrameRate")
    if node and genicam.IsWritable(node):
        node.SetValue(5.0)
        print("Set AcquisitionFrameRate = 5")

    # Other cameras use AcquisitionFrameRateAbs
    node = get_node(nodemap, "AcquisitionFrameRateAbs")
    if node and genicam.IsWritable(node):
        node.SetValue(5.0)
        print("Set AcquisitionFrameRateAbs = 5")

    camera.StartGrabbingMax(10)

    while camera.IsGrabbing():
        result = camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )
        try:
            print("Grab success:", result.GrabSucceeded())
            if not result.GrabSucceeded():
                print(result.ErrorCode, result.ErrorDescription)
        finally:
            result.Release()

finally:
    if camera.IsGrabbing():
        camera.StopGrabbing()
    if camera.IsOpen():
        camera.Close()
