/*
 Face Detection and left eye tracking
 */
using UnityEngine;
using UnityEngine.UI;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.OCR;

public class FaceDetector : MonoBehaviour
{
    VideoCapture video;

    // Use the LBP Cascade for frontal face
    private CascadeClassifier frontalFace;
    private string pathToFrontalCascadeClassifier = "Assets/ClassifiersXML/lbpcascade_frontalface_improved.xml";
    private Vector3 pov;
    public Vector2 eyeShift;

    void Start()
    {
        video = new VideoCapture(0);
        if (video.IsOpened)
            video.ImageGrabbed += OnImageGrabbed;
    }

    // Function for grabbing images
    void OnImageGrabbed(object sender, EventArgs args)
    {        
        Mat source = new Mat();
        video.Retrieve(source);

        // Crop image to get rid of black stripes
        Image<Bgr, Byte> buffer_im = source.ToImage<Bgr, Byte>();
        buffer_im.ROI = new Rectangle(new Point(0, 60), new Size(source.Width, source.Height - 60));
        Image<Bgr, Byte> cropped_im = buffer_im.Copy();
        source = cropped_im.Mat;

        // Flip the image
        CvInvoke.Flip(source, source, FlipType.Horizontal);

        // Convert to gray image to use the cascade classifier
        Mat imgGray = source.Clone();
        CvInvoke.CvtColor(source, imgGray, ColorConversion.Bgr2Gray);

        frontalFace = new CascadeClassifier(pathToFrontalCascadeClassifier);
        Rectangle[] detectFaces = frontalFace.DetectMultiScale(imgGray);
        if (detectFaces.Length == 0)
        {
            return;
        }

        // Keep the biggest face
        Rectangle mainFace = detectFaces[0];
        foreach (var face in detectFaces)
        {
            if (face.Height * face.Width > mainFace.Height * face.Width)
            {
                mainFace = face;
            }
        }

        // Draw a rectangle around the detected face
        CvInvoke.Rectangle(source, mainFace, new MCvScalar(0, 255, 0));
        
        // Find the left eye for the point of view
        PointF eyes = new PointF((float) mainFace.Left + (float) mainFace.Width / eyeShift.x, (float) mainFace.Top + (float) mainFace.Height / eyeShift.y /* - 60.0f*/);
        // Draw a circle around the left eye
        CvInvoke.Circle(source, Point.Round(eyes), 3, new MCvScalar(0, 0, 255), 2);

        // Find the screen width and height
        float screenWidth = GetComponent<AsymFrustum>().width;
        float screenHeight = GetComponent<AsymFrustum>().width;

        // Convert left eye position to camera position
        pov = new Vector3(((eyes.X / video.Width) - 0.5f) * screenWidth, -((eyes.Y / (video.Height-120.0f)) - 0.5f) * screenHeight, transform.position.z);
        transform.position = pov;

        CvInvoke.Imshow("face Detection", source);
    }

    void Update()
    {
        if (video.IsOpened)
            video.Grab();
    }
}
