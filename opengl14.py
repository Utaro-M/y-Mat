#-*- coding: utf-8 -*-
#目の位置のブレを減らすため、かおの位置との相対位置を利用
#テクスチャーの貼り付け
import cv2
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import random
import math
import sys
import threading
import time
import numpy as np
from PIL import Image,ImageDraw, ImageFilter,ImageOps

#smile検出、０，１
smile=0

# init position
init=math.pi/3.0
th=init
ph=init

#拡大、縮小
r=10
#mouse move
mx = 0.0
my = 0.0

c1=0
c2=0
mipmap=0

img=None
c_img=0

tex=0
mipmap=0


def main():
    thread1=threading.Thread(target=gl)
    thread2=threading.Thread(target=cv)

    thread1.start()
    thread2.start()

'''
def cv():
    global angy,angz
    i=0
    while (i<10):
        time.sleep(0.5)
        angy+=10
        angz+=10
        i+=1
        print("angy=",angy)
'''

def cv():

    global th,ph,smile,c1,c2,ppm_path,tex,mipmap,img,c_img
    omega=math.pi/7
    threshold=30
    cap = cv2.VideoCapture(0)
    img_last = None # 前回の画像を記憶する変数 --- (*1)
    eyes_ave_last=None
    green = (0, 255, 0)
    eye_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_eye.xml")
    faceCascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"    )
    smile_cascade = cv2.CascadeClassifier("/usr/share/opencv/haarcascades/haarcascade_smile.xml")
    while True:
        time.sleep(0.05)
        #画像を処理
        _,frame = cap.read()
        frame   = cv2.resize(frame,(800,600)) #500,300


        #白黒に変換
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray,(9,9),0)
        img_b = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)[1]

        #目の検出

        eyes = eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(2, 2),
        )
        for i in range(len(eyes)-1):
            if eyes[i][1]-eyes[i+1][1]<30:
                eye1=eyes[i]
                eye2=eyes[i+1]

        #両目の平均
        eyes_ave=[0,0,0,0]
        for j in range(4):
            eyes_ave[j] = (eye1[j]+eye2[j]) /2 #int or float
            [ex,ey,ew,eh] = eyes_ave

        if ew<30:continue
        cv2.circle(frame,(ex,ey),30,(250,0,0),4,4)

        #かおの検出

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(20, 20)
        )

        face=[0,0,0,0]
        #かおの表示
        for (x,y,w,h) in faces:
            #if abs(x-ex)<15 and abs(y-ey)<15: continue
            face[0]=x
            face[1]=y
            face[2]=w
            face[3]=h
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
        [fx,fy,fw,fh]=face
        #かおの切り取り



        if c_img==0:
            img = frame[fy:fy+fh,fx:fx+fw]
            img=cv2.resize(img,(256,256))
            b,g,r = cv2.split(img)
            #cv2.flip(img,-1)
            cv2.imwrite("face.jpg",img)
            c_img+=1
            im1 = Image.open('./sora2.jpg')
            im2 = Image.open('./face.jpg')
            mask_im = Image.new("L", im2.size, 0)
            draw = ImageDraw.Draw(mask_im)
            draw.ellipse((5, 5, 245, 245), fill=255)
            mask_im_blur = mask_im.filter(ImageFilter.GaussianBlur(10))
            im1.paste(im2, (0,0), mask_im_blur)
            #im1=ImageOps.flip(im1)
            im1=ImageOps.mirror(im1)
            #im1=ImageOps.mirror(im1)
            im1.save("./face_circle.jpg",quality=95)
        '''
        if c_img==0:
            img = frame[fy:fy+fh,fx:fx+fw]
            img=cv2.resize(img,(256,256))
            cv2.flip(img,-1)
            cv2.imwrite("face.jpg", img)
            c_img+=1
            img_original = cv2.imread("./sora2.jpg")
            original_h, original_w = img_original.shape[:2]
            face_img = cv2.imread("./face.jpg")
            face_h, face_w = face_img.shape[:2]
            img_original[10:face_h + 10, 10:face_h + 10] = face_img
            cv2.imwrite("sora2_face.jpg",img_original)

        '''
        #かおとの相対位置が適切な場合はph,thを変化させる（適切なとき、黒丸が表示される）うまくいっていない
        if ey>fy and ey<(fy+0.5*fh):
            cv2.circle(frame,(fx,fy),10,(0,0,0),2,2)


            #前回の目の位置を取ってくる
            if eyes_ave_last==None:
                eyes_ave_last=eyes_ave
            [epx,epy,epw,eph] = eyes_ave_last

            dltax=ex-epx
            dltay=ey-epy


            #gl()へph,thを送る
            '''
            if abs(dltax)>100 or abs(dltay)>100:
                ph=ph
                th=th
            el
            '''
            if dltax<(-threshold):
                if dltay<(-threshold):
                    ph-=omega
                    th-=omega
                elif abs(dltay)<threshold:
                    ph-=omega
                    th=th
                else:
                    ph-=omega
                    th+=omega
            elif abs(dltax)<threshold:
                if dltay<(-threshold):
                    ph=ph
                    th-=omega
                elif abs(dltay)<threshold:
                    ph=ph
                    th=th
                else:
                    ph=ph
                    th+=omega
            else:
                if dltay<(-threshold):
                    ph+=omega
                    th-=omega
                elif abs(dltay)<threshold:
                    ph+=omega
                    th=th
                else:
                    ph+=omega
                    th+=omega
        else:
            ph=ph
            th=th

        #笑顔の検出
        smiles= smile_cascade.detectMultiScale(roi_gray,scaleFactor= 1.2, minNeighbors=9, minSize=(20, 20))#笑顔識別
        if len(smiles) ==0 :
            smile=0
            print ("No smile")
        else:
            smile=1
            print ("smile")
        '''
        if smile==0:
            c1=c1+1
            c2=0
            if c1>=5 :
                ppm_path = os.path.join("ame.jpg")
                print ppm_path
            else:
                ppm_path = os.path.join("sora2.jpg")

        else:
            c2=c2+1
            c1=0
            if c2>=5:
                ppm_path = os.path.join("sora2.jpg")
            else:
                ppm_path = os.path.join("ame.jpg")
        '''

        # 今回のフレームを保存 --- (*5)
        img_last = img_b
        eyes_ave_last=eyes_ave
        # 画面に表示
        cv2.imshow("Diff Camera", frame)
        #cv2.imshow("diff data", frame_diff)

        if cv2.waitKey(1) == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()



def gl():
    global tex1,tex2
    glutInitWindowPosition(600, 100);
    glutInitWindowSize(600, 600);
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE )
    glutCreateWindow("pyOpenGL TEST")
    glutDisplayFunc(draw)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)

    #ppm_path = os.path.join("sora2.jpg")
    tex1 = ppm2texture(os.path.join("face_circle.jpg"))
    #mipmap = ppm2mipmap(ppm_path)
    tex2 = ppm2texture(os.path.join("sora2.jpg"))



    '''
    #glutMotionFunc(motion)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    '''
    init()
    glutIdleFunc(idle)
    glutMainLoop()

def ppm2texture(ppm_path):
    ppm = Image.open(ppm_path)
    # assert check_size(ppm)
    w, h = ppm.size
    # data = ppm.tostring()
    data = ppm.tobytes()

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h,
            0, GL_RGB, GL_UNSIGNED_BYTE, data)
    return tex

def ppm2mipmap(ppm_path):
    ppm = Image.open(ppm_path)
    # assert check_size(ppm)
    w, h = ppm.size
    # data = ppm.tostring()
    data = ppm.tobytes()

    tex = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, mipmap)
    gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGB, w, h,
                      GL_RGB, GL_UNSIGNED_BYTE, data)
    return tex


def draw():
    global angy,angz,th,ph,c1,c2,smile,r,tex1,tex2
    cx = cy = cz =  0.0
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    glClearColor(0.0, 0.0, 0.0, 1.0)

    glLoadIdentity()
    if smile==0:
        c1=c1+1
        c2=0
        if c1>=15 :
            glClearColor(0.6,0.6,0.6, 1.0)

        else:
            glClearColor(1.0,1.0,1.0, 1.0)

    else:
        c2=c2+1
        c1=0
        if c2>=10:
            glClearColor(1.0,1.0,1.0, 1.0)
        else:
            glClearColor(0.6,0.6,0.6, 1.0)

    glColor3f(1.0,1.0,1.0)
    '''
    # ppm_path = os.path.join(os.path.dirname(__file__), u"texture2.ppm")
    if smile==0:
        c1=c1+1
        c2=0
        if c1>=100 :
            ppm_path = os.path.join("ame.jpg")
        else:
            ppm_path = os.path.join("sora2.jpg")

    else:
        c2=c2+1
        c1=0
        if c2>=100:
            ppm_path = os.path.join("sora2.jpg")
        else:
            ppm_path = os.path.join("ame.jpg")
    '''


    '''

    tex = ppm2texture(ppm_path)
    mipmap = ppm2mipmap(ppm_path)
    #glutMotionFunc(motion)
    glEnable(GL_TEXTURE_2D)
    glBindTexture(GL_TEXTURE_2D, tex)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    '''

    gluLookAt( r*math.sin(th)*math.cos(ph)+cx,  r*math.cos(th)+cy,  r*math.sin(th)*math.sin(ph)+cz,\
            cx,  cy,  cz, -math.cos(th)*math.cos(ph),math.sin(th),-math.cos(th)*math.sin(ph) )

    # テクスチャマップを有効にする
    glEnable(GL_TEXTURE_2D)
    # 1 つ目のテクスチャを設定
    glBindTexture(GL_TEXTURE_2D, tex1)
    # テクスチャマップの方法を設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)


    glBegin(GL_QUADS)

    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, -1.0, -1.0)
    glEnd()

    # 2つ目のテクスチャを設定
    glBindTexture(GL_TEXTURE_2D, tex2)
    # テクスチャマップの方法を設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    '''
    # 2 つ目のテクスチャを設定
    glBindTexture(GL_TEXTURE_2D, mipmap)
    # テクスチャマップの方法を設定
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
                    GL_LINEAR_MIPMAP_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    '''
    glBegin(GL_QUADS)

    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0,  1.0,  1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0,  1.0,  1.0)


    glTexCoord2f(0.0, 1.0)
    glVertex3f(-1.0,  1.0, -1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(-1.0,  1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(1.0,  1.0,  1.0)
    glTexCoord2f(1.0, 1.0)
    glVertex3f(1.0,  1.0, -1.0)

    glTexCoord2f(1.0, 1.0)
    glVertex3f(-1.0, -1.0, -1.0)
    glTexCoord2f(0.0, 1.0)
    glVertex3f(1.0, -1.0, -1.0)
    glTexCoord2f(0.0, 0.0)
    glVertex3f(1.0, -1.0,  1.0)
    glTexCoord2f(1.0, 0.0)
    glVertex3f(-1.0, -1.0,  1.0)

    glEnd()

    glutSwapBuffers()


def init():
    glClearColor(0.7, 0.7, 0.7, 0.7)
    # initialize texture mapping


def idle():

    glutPostRedisplay()


def reshape(w, h):
    glViewport(0, 0,w,h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(30.0, w/h, 1.0, 100.0)
    glMatrixMode (GL_MODELVIEW)


def keyboard(key,x,y):
    global ph,th,r
    if key=="r":
        ph=math.pi/3
        th=math.pi/3
        r=10
    #12段階で拡大縮小
    if key=="b":
        r-=1
        if r<=4:
            r=4
    if key=="s":
        r+=1
        if r>=15:
            r=15



'''
def loadImage():
      image = open("sora2.jpg")

      ix = image.size[0]
      iy = image.size[1]
      image = image.tostring("raw", "RGBX", 0, -1)

      glPixelStorei(GL_UNPACK_ALIGNMENT,1)
      glTexImage2D(GL_TEXTURE_2D, 0, 3, ix, iy, 0, GL_RGBA, GL_UNSIGNED_BYTE, image)
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP)
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP)
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
      glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
      glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)
'''

if __name__ == "__main__":
    main()
#立方体の描画 http://aidiary.hatenablog.com/entry/20080831/1281750119
#テクスチャ貼り https://fgshun.hatenablog.com/entry/20080922/1222095288
#精度向上　 https://qiita.com/FukuharaYohei/items/116932920c99a5b73b32
#笑顔認識　　https://qiita.com/fujino-fpu/items/99ce52950f4554fbc17d
#画像トリミング  https://qiita.com/Dexctersu/items/f193771afba3f5f1bd2e
#Python Pillow   https://note.nkmk.me/python-pillow-paste/
