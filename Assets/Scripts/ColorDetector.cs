/*
 Color Detection and centre tracking of the object
 For this example, it was to track the red smartphone for the smartphone camera's perspective
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

public class ColorDetector : MonoBehaviour
{
    VideoCapture video;

    // HSV 1 of Red color
    public Vector3 hsv1Min = new Vector3(0, 120, 70);
    public Vector3 hsv1Max = new Vector3(10, 255, 255);

    // HSV 2 of Red color
    public Vector3 hsv2Min = new Vector3(170, 120, 70);
    public Vector3 hsv2Max = new Vector3(180, 255, 255);

    // Point of view
    private Vector3 pov;

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

        // Detect the Red color in the image
        Mat final = ImageTreatment(source);

        // Find the centroid of the biggest contour
        Vector2 eye = Borders(final, source);

        // Find the screen width and height
        float screenWidth = GetComponent<AsymFrustum>().width;
        float screenHeight = GetComponent<AsymFrustum>().width;

        // Convert eye position (Centroid) to camera position
        pov = new Vector3(((eye.x / video.Width) - 0.5f) * screenWidth, -((eye.y / (video.Height - 120.0f)) - 0.5f) * screenHeight, transform.position.z);
        transform.position = pov;

        CvInvoke.Imshow("Color Detection", source);
    }

    // Function for image treatment
    Mat ImageTreatment(Mat image)
    {
        CvInvoke.CvtColor(image, image, ColorConversion.Bgr2Hsv);
        // Use Median filter
        CvInvoke.MedianBlur(image, image, 3);

        // Prepare the lower and higher threshold and then use it to crop the image
        Hsv lower = new Hsv(hsv1Min.x, hsv1Min.y, hsv1Min.z);
        Hsv higher = new Hsv(hsv1Max.x, hsv1Max.y, hsv1Max.z);

        Image<Hsv, Byte> i = image.ToImage<Hsv, Byte>();
        Mat result1 = i.InRange(lower, higher).Mat;

        lower = new Hsv(hsv2Min.x, hsv2Min.y, hsv2Min.z);
        higher = new Hsv(hsv2Max.x, hsv2Max.y, hsv2Max.z);

        Mat result2 = i.InRange(lower, higher).Mat;

        Mat result = result1 + result2;

        int operationSize = 1;

        // Use the opening (Erode then Dilate)
        Mat structuringElement = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                                                                new Size(2 * operationSize + 1, 2 * operationSize + 1),
                                                                new Point(operationSize, operationSize));

        CvInvoke.Erode(result, result, structuringElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        CvInvoke.Dilate(result, result, structuringElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

        return result;
    }

    // Function to find the centre of the biggest object's borders found in the binary image resulted from the imageTreatment function
    Vector2 Borders(Mat image, Mat origin)
    {
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        VectorOfPoint biggestContour = new VectorOfPoint();
        int biggestContourIndex = -1;
        double biggestContourArea = 0;

        // Find contours of the image
        Mat hierarchy = new Mat();
        CvInvoke.FindContours(image, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxNone);

        // Find the biggest contour
        for (int i = 0; i < contours.Size; i++)
        {
            if (CvInvoke.ContourArea(contours[i]) > biggestContourArea)
            {
                biggestContour = contours[i];
                biggestContourIndex = i;
                biggestContourArea = CvInvoke.ContourArea(contours[i]);
            }
        }

        // Draw the biggest contour
        if (biggestContourIndex > -1)
            CvInvoke.DrawContours(origin, contours, biggestContourIndex, new MCvScalar(0, 0, 255), 2);

        // Calculate the centroid of the biggest contour
        var moments = CvInvoke.Moments(image);
        var Centroid = new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));
        CvInvoke.Circle(origin, Centroid, 2, new MCvScalar(0, 0, 255), 2);

        return new Vector2(Centroid.X, Centroid.Y);
    }

    void Update()
    {
        if (video.IsOpened)
            video.Grab();
    }
}
