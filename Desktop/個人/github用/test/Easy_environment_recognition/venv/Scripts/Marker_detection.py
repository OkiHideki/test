import numpy as np
import cv2
from cv2 import aruco
from HandTracking import Hand_Tracking

def coordinate_convert(org_x, org_y, width, height):
    x = org_x - (width / 2)
    y = height / 2 - org_y
    return x, y

def main():
    cap = cv2.VideoCapture(1)

    # カメラの解像度
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print("width : ", width)
    print("height : ",  height)
    # マーカーサイズ
    marker_length = 0.080 # 一辺[m]
    # マーカーの辞書選択
    dictionary_name = aruco.DICT_6X6_250
    dictionary = aruco.getPredefinedDictionary(dictionary_name)

    camera_matrix = np.load("./calibration_data/mtx.npy")
    distortion_coeff = np.load("./calibration_data/dist.npy")

    while True:
        ret, img = cap.read()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(img, dictionary)
        img, dst = Hand_Tracking(img)
        if dst:
            print("dst : ", dst) # 画素間でしかないので、現実の距離に変換する必要あり

        # 可視化
        aruco.drawDetectedMarkers(img, corners, ids, (0,255,255))

        if len(corners) > 0:
            # マーカーごとに処理
            for i, corner in enumerate(corners):
                # rvec -> rotation vector, tvec -> translation vector
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)

                # < rodoriguesからeuluerへの変換 >

                # 不要なaxisを除去
                tvec = np.squeeze(tvec)
                rvec = np.squeeze(rvec)
                # 回転ベクトルからrodoriguesへ変換
                rvec_matrix = cv2.Rodrigues(rvec)
                rvec_matrix = rvec_matrix[0] # rodoriguesから抜き出し
                # 並進ベクトルの転置(.T)
                transpose_tvec = tvec[np.newaxis, :].T
                # 合成
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                # オイラー角への変換
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]

                # 画像の中心を座標軸の原点に変換
                center = np.mean(corner[0], axis=0)
                conv_center = coordinate_convert(center[0], center[1], width, height)

                print("ids : ", ids[i])
                print("coordinate : ", conv_center)
                print("z : " + str(tvec[2]))
                '''# m単位で計算するのは下記
                print("x : " + str(tvec[0]))
                print("y : " + str(tvec[1]))
                '''
                print("roll : " + str(180 + euler_angle[0]))
                print("pitch: " + str(euler_angle[1]))
                print("yaw  : " + str(0 - euler_angle[2]))

                # 可視化
                draw_pole_length = marker_length/2 # 現実での長さ[m]
                aruco.drawAxis(img, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)

        cv2.imshow('drawDetectedMarkers', img)
        k = cv2.waitKey(10)
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
