using UnityEngine;

using System;
using System.Collections;
using System.Collections.Generic;
using System.Drawing;
using System.IO;

using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.Features2D;

public class Essai : MonoBehaviour
{
    VideoCapture video;
    public Vector3 hsvMin = new Vector3(20, 100, 100);
    public Vector3 hsvMax = new Vector3(30, 255, 255);

    [Range (0, 50)]
    public float threshold = 2;

    // Start is called before the first frame update
    void Start()
    {
        video = new VideoCapture(0);
    }

    // Update is called once per frame
    void Update()
    {
        Mat orig = new Mat();

        if (video.IsOpened)
            orig = video.QueryFrame();

        if (orig.IsEmpty)
            return;

        if (orig != null)
        {
            CvInvoke.Flip(orig, orig, FlipType.Horizontal);
            CvInvoke.Imshow("Webcam View", orig);
            CvInvoke.WaitKey(24);

            Mat image2 = orig.Clone();
            Mat final = ImageTreatment(image2);
            CvInvoke.Imshow("Webcam HSV", final);

            //Mat mul = orig & final; // Image<Bgr, Byte> mul = final.ToImage<Bgr, Byte>();
            //CvInvoke.Multiply(orig, final, mul);
            //CvInvoke.Imshow("Multiplication", mul);
            Image<Gray, Byte> image3 = CornersDetector(orig.ToImage<Gray, Byte>(), final.ToImage<Gray, Byte>());
            Mat borders = Borders(final, orig);
            CvInvoke.Imshow("Corner Detection", borders);
        }
        else
            CvInvoke.DestroyAllWindows();
    }
    void OnDestroy()
    {
        CvInvoke.DestroyAllWindows();
    }

    Mat ImageTreatment(Mat image)
    {
        CvInvoke.CvtColor(image, image, ColorConversion.Bgr2Hsv);
        CvInvoke.MedianBlur(image, image, 3);


        Hsv lower = new Hsv(hsvMin.x, hsvMin.y, hsvMin.z);
        Hsv higher = new Hsv(hsvMax.x, hsvMax.y, hsvMax.z);

        Image<Hsv, Byte> i = image.ToImage<Hsv, Byte>();
        Mat result = i.InRange(lower, higher).Mat;

        int operationSize = 1;

        Mat structuringElement = CvInvoke.GetStructuringElement(ElementShape.Ellipse,
                                                                new Size(2 * operationSize + 1, 2 * operationSize + 1),
                                                                new Point(operationSize, operationSize));

        CvInvoke.Erode(result, result, structuringElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));
        CvInvoke.Dilate(result, result, structuringElement, new Point(-1, -1), 1, BorderType.Default, new MCvScalar(0));

        return result;
    }

    Mat Borders(Mat image, Mat origin)
    {
        VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint();
        VectorOfPoint biggestContour = new VectorOfPoint();
        int biggestContourIndex = -1;
        double biggestContourArea = 0;

        Mat hierarchy = new Mat();
        CvInvoke.FindContours(image, contours, hierarchy, RetrType.List, ChainApproxMethod.ChainApproxNone);

        for (int i = 0; i < contours.Size; i++)
        {
            if (CvInvoke.ContourArea(contours[i]) > biggestContourArea)
            {
                biggestContour = contours[i];
                biggestContourIndex = i;
                biggestContourArea = CvInvoke.ContourArea(contours[i]);
            }
        }
        
        if (biggestContourIndex > -1)
        {
            VectorOfPoint corners = new VectorOfPoint();
            //.01 * CvInvoke.ArcLength(biggestContour, true)
            CvInvoke.ApproxPolyDP(biggestContour, corners, .005 * CvInvoke.ArcLength(biggestContour, true), true);
            //for (int i = 0; i < corners.Size; i++)
            //{
            //    CvInvoke.Circle(origin, corners[i], 2, new MCvScalar(255, 255, 0), 2);
            //}

            List<Point> listCorners = new List<Point>(corners.ToArray());

            float minDist = 0;
            while (minDist < threshold)
            {
                minDist = 1e9f;
                int minIndex = 0;
                for (int did = 0; did < listCorners.Count; ++did)
                {
                    Point p = listCorners[did];
                    Point a = did == 0 ? listCorners[listCorners.Count - 1] : listCorners[did - 1];
                    Point b = did == listCorners.Count - 1 ? listCorners[0] : listCorners[did + 1];

                    float dist = PointToLineDistance(p, a, b);
                    if (dist < minDist)
                    {
                        minDist = dist;
                        minIndex = did;
                    }
                }

                listCorners.RemoveAt(minIndex);
            }

            for (int i = 0; i < listCorners.Count; i++)
            {
                CvInvoke.Circle(origin, listCorners[i], 2, new MCvScalar(0, 255, 0), 2);
            }



        }
        //CvInvoke.DrawContours(origin, contours, biggestContourIndex, new MCvScalar(0, 0, 255), 2);

        var moments = CvInvoke.Moments(image);
        var Centroid = new Point((int)(moments.M10 / moments.M00), (int)(moments.M01 / moments.M00));
        CvInvoke.Circle(origin, Centroid, 2, new MCvScalar(0, 0, 255), 2);
        //Debug.Log(Centroid);

        return origin;
    }

    float PointToLineDistance (Point p, Point a, Point b)
    {
        float f = Math.Abs((b.Y - a.Y) * p.X - (b.X - a.X) * p.Y + b.X * a.Y - b.Y * a.X);
        float g = (float) Math.Sqrt((b.Y - a.Y) * (b.Y - a.Y) + (b.X - a.X) * (b.X - a.X));

        return f / g;
    }

    Image<Gray, Byte> CornersDetector (Image<Gray, Byte> image, Image<Gray, Byte> binaryImage)
    {
        Image<Gray, Byte> final = image.Clone();
        CvInvoke.CornerHarris(binaryImage, final, 3);
        return final;
    }
}
