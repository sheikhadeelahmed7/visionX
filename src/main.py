from FaceRegistration import FaceRegistrar

IMG_PATH = "PATH TO BE CONF"

if __name__ == '__main__':
    face_rg = FaceRegistrar()
    status , statusBar = face_rg.getstatus(IMG_PATH)

    print('STOP')
