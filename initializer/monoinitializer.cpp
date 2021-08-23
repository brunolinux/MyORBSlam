#include "monoinitializer.h"
#include "keypointmatcher.h"
#include "Random.h"
#include <thread>

using std::vector;
using std::thread;

MonoInitializer::MonoInitializer(int max_iterations, float sigma)
    :m_state(InitState::NOInit), m_maxIterations(max_iterations), m_sigma(sigma)
{}

bool MonoInitializer::initialize(std::shared_ptr<Frame> frame)
{
    if (m_state == InitState::NOInit) {
        if (checkFrame(frame)) {
            m_reference = frame;
            m_state = InitState::ReferencePassed;
        }

        return false;
    } else if (m_state == InitState::ReferencePassed) {
        if (checkFrame(frame)) {
            m_target = frame;

            // Find correspondences
            KeypointMatcher matcher(0.9, true);
            m_matches = matcher.SearchForInitialization(m_reference.get(), m_target.get(), 100);

            if (checkFrameMatching(m_matches)) {
                if (constructPose()) {
                    m_state = InitState::TargetPassed;
                    return true;
                }
            }
        }

        m_state = InitState::NOInit;
        return false;
    } else {
        return true;
    }
}

bool MonoInitializer::constructPose()
{
    m_indexesGroups = constructConstantNumberElementsGroups(m_matches.size(), m_maxIterations, MIN_POINTSET_NUM);

    // Launch threads to compute in parallel a fundamental matrix and a homography
    vector<bool> inlier_matches_H, inlier_mactches_F;
    float score_H, score_F;
    cv::Mat H, F;

    thread threadH(&MonoInitializer::findHomography, this, std::ref(inlier_matches_H), std::ref(score_H), std::ref(H));
    thread threadF(&MonoInitializer::findFundamental, this, std::ref(inlier_mactches_F), std::ref(score_F), std::ref(F));

    // Wait until both threads have finished
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    float RH = score_H/(score_H + score_F);

    cv::Mat K = Frame::getIntrinsicMatrix();
    std::vector<cv::Point3f> points_3d;
    std::vector<bool> triangulated_states;
    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    if (RH > 0.40)
        return reconstructH(inlier_matches_H, H, K, Rcw, tcw, points_3d, triangulated_states, 1.0, 50);
    else //if(pF_HF>0.6)
        return reconstructF(inlier_mactches_F,F, K, Rcw, tcw, points_3d, triangulated_states, 1.0, 50);

}

void MonoInitializer::findHomography(std::vector<bool> &inlier_matches, float &score, cv::Mat &H21)
{
    // Number of putative matches
    const int N = m_matches.size();

    // Normalize coordinates
    vector<cv::Point2f> points1, points2;
    cv::Mat T1, T2;
    normalizePoints(m_reference->getUndistortedKeypoints(), points1, T1);
    normalizePoints(m_target->getUndistortedKeypoints(), points2, T2);
    cv::Mat T2inv = T2.inv();

    // Iteration variables
    vector<cv::Point2f> points1_subset(MIN_POINTSET_NUM);
    vector<cv::Point2f> points2_subset(MIN_POINTSET_NUM);
    cv::Mat H21i, H12i;
    vector<bool> current_inliers(N, false);
    float current_score;

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < m_maxIterations; it++) {
        // Select a minimum set
        for (size_t j = 0; j < MIN_POINTSET_NUM; j++) {
            int idx = m_indexesGroups[it][j];

            points1_subset[j] = points1[m_matches[idx].src_idx];
            points2_subset[j] = points2[m_matches[idx].dst_idx];
        }

        cv::Mat Hn = computeH21(points1_subset, points2_subset);
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        current_score = checkHomography(H21i, H12i, current_inliers, m_sigma);

        if(current_score > score) {
            H21 = H21i.clone();
            inlier_matches = current_inliers;
            score = current_score;
        }
    }
}

float MonoInitializer::checkHomography(const cv::Mat &H21, const cv::Mat &H12, std::vector<bool> &inlier_matches, float sigma)
{
    const int N = m_matches.size();

    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    const float th = 5.991;
    const float invSigmaSquare = 1.0/(sigma*sigma);


    inlier_matches.resize(N);
    float score = 0;
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = m_reference->getUndistortedKeypoints()[m_matches[i].src_idx];
        const cv::KeyPoint &kp2 = m_target->getUndistortedKeypoints()[m_matches[i].dst_idx];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2

        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1 > th)
            bIn = false;
        else
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1

        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2 > th)
            bIn = false;
        else
            score += th - chiSquare2;

        if(bIn)
            inlier_matches[i]=true;
        else
            inlier_matches[i]=false;
    }

    return score;
}

void MonoInitializer::findFundamental(std::vector<bool> &inlier_matches, float &score, cv::Mat &F21)
{
    // Number of putative matches
    const int N = m_matches.size();

    // Normalize coordinates
    vector<cv::Point2f> points1, points2;
    cv::Mat T1, T2;
    normalizePoints(m_reference->getUndistortedKeypoints(), points1, T1);
    normalizePoints(m_target->getUndistortedKeypoints(), points2, T2);
    cv::Mat T2inv = T2.inv();

    // Iteration variables
    vector<cv::Point2f> points1_subset(MIN_POINTSET_NUM);
    vector<cv::Point2f> points2_subset(MIN_POINTSET_NUM);
    cv::Mat F21i;
    vector<bool> current_inliers(N, false);
    float current_score;

    // Perform all RANSAC iterations and save the solution with highest score
    for (int it = 0; it < m_maxIterations; it++) {
        // Select a minimum set
        for (size_t j = 0; j < MIN_POINTSET_NUM; j++) {
            int idx = m_indexesGroups[it][j];

            points1_subset[j] = points1[m_matches[idx].src_idx];
            points2_subset[j] = points2[m_matches[idx].dst_idx];
        }

        cv::Mat Fn = computeF21(points1_subset, points2_subset);
        F21i = T2inv*Fn*T1;

        current_score = checkFundamental(F21i, current_inliers, m_sigma);

        if(current_score > score) {
            F21 = F21i.clone();
            inlier_matches = current_inliers;
            score = current_score;
        }
    }
}

float MonoInitializer::checkFundamental(const cv::Mat &F21, std::vector<bool> &inlier_matches, float sigma)
{
    const int N = m_matches.size();

    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    inlier_matches.resize(N);

    float score = 0;

    const float th = 3.841;
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        const cv::KeyPoint &kp1 = m_reference->getUndistortedKeypoints()[m_matches[i].src_idx];
        const cv::KeyPoint &kp2 = m_target->getUndistortedKeypoints()[m_matches[i].dst_idx];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)

        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1 > th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)

        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2 > th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        if(bIn)
            inlier_matches[i]=true;
        else
            inlier_matches[i]=false;
    }

    return score;
}

bool MonoInitializer::reconstructH(const std::vector<bool> &inlier_matches,
                                   const cv::Mat &H21, const cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21,
                                   std::vector<cv::Point3f> &points_3d, std::vector<bool> &triangulated_states,
                                   float minParallax, int minTriangulated)
{
    int N = 0;
    for(auto match : inlier_matches) {
        if(match) {
            N++;
        }
    }

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    V=Vt.t();

    float s = cv::determinant(U)*cv::determinant(Vt);

    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    if(d1/d2 < 1.00001 || d2/d3 < 1.00001) {
        return false;
    }

    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    //case d'=d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    int bestGood = 0;
    int secondBestGood = 0;
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    for(size_t i = 0; i < 8; i++)
    {
        float parallaxi;
        vector<cv::Point3f> vP3Di;
        vector<bool> vbTriangulatedi;
        int nGood = checkRt(vR[i],vt[i], inlier_matches, K, 4.0*m_sigma*m_sigma, vP3Di, vbTriangulatedi, parallaxi);

        if (nGood > bestGood) {
            secondBestGood = bestGood;
            bestGood = nGood;
            bestSolutionIdx = i;
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        } else if (nGood > secondBestGood) {
            secondBestGood = nGood;
        }
    }

    if(secondBestGood<0.75*bestGood && bestParallax>=minParallax &&
       bestGood>minTriangulated && bestGood>0.9*N) {
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        points_3d = bestP3D;
        triangulated_states = bestTriangulated;

        return true;
    } else {
        return false;
    }
}

bool MonoInitializer::reconstructF(const std::vector<bool> &inlier_matches,
                                   const cv::Mat &F21, const cv::Mat &K,
                                   cv::Mat &R21, cv::Mat &t21,
                                   std::vector<cv::Point3f> &points_3d, std::vector<bool> &triangulated_states,
                                   float minParallax, int minTriangulated)
{
    int N = 0;
    for(auto match : inlier_matches) {
        if(match) {
            N++;
        }
    }

    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    decomposeEssentialMatrix(E21, R1, R2, t);

    cv::Mat t1 = t;
    cv::Mat t2 = -t;

    // Reconstruct with the 4 hyphoteses and check
    vector<vector<cv::Point3f>> vP3D_set(4);
    vector<vector<bool>> vbTriangulated_set(4);
    vector<float> parallax_set(4);
    vector<int> ngood_set(4);

    vector<cv::Mat> rotation_set{R1, R2, R1, R2};
    vector<cv::Mat> translation_set{t1, t1, t2, t2};

    float threshold = 4.0 * m_sigma * m_sigma;
    for (size_t i = 0; i < rotation_set.size(); i ++) {
        ngood_set[i] = checkRt(rotation_set[i], translation_set[i], inlier_matches, K, threshold,
                               vP3D_set[i], vbTriangulated_set[i], parallax_set[i]);
    }

    auto max_iter = std::max(ngood_set.begin(), ngood_set.end());
    int maxGood = *max_iter;
    int maxIndex = max_iter - ngood_set.begin();

    int nsimilar = 0;
    for (size_t i = 0; i < ngood_set.size(); i ++) {
        if (ngood_set[i] > 0.7 * maxGood) {
            nsimilar ++;
        }
    }

    int nMinGood = std::max(static_cast<int>(0.9*N), minTriangulated);
    // If there is not a clear winner or not enough triangulated points reject initialization
    if(maxGood < nMinGood || nsimilar > 1) {
        return false;
    }

    R21 = cv::Mat();
    t21 = cv::Mat();
    if (parallax_set[maxIndex] > minParallax) {
        points_3d = std::move(vP3D_set[maxIndex]);
        triangulated_states = std::move(vbTriangulated_set[maxIndex]);

        rotation_set[maxIndex].copyTo(R21);
        translation_set[maxIndex].copyTo(t21);
        return true;
    } else {
        return false;
    }
}

int MonoInitializer::checkRt(const cv::Mat &R, const cv::Mat &t,
                             const std::vector<bool> &inlier_matches, const cv::Mat &K, float threshold,
                             std::vector<cv::Point3f> &points_3d, std::vector<bool> &triangulated_states, float &parallax)
{
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    size_t reference_kp_size = m_reference->getKeypointSize();
    triangulated_states = vector<bool>(reference_kp_size, false);
    points_3d.resize(reference_kp_size);

    vector<float> vCosParallax;
    vCosParallax.reserve(reference_kp_size);

    // Camera 1 Projection Matrix K[I|0]
    cv::Mat P1(3,4,CV_32F,cv::Scalar(0));
    K.copyTo(P1.rowRange(0,3).colRange(0,3));

    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    cv::Mat P2(3, 4, CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    cv::Mat O2 = -R.t()*t;

    int nGood=0;

    for(size_t i = 0; i < m_matches.size(); i++) {
        if (!inlier_matches[i])
            continue;

        const cv::KeyPoint &kp1 = m_reference->getUndistortedKeypoints()[m_matches[i].src_idx];
        const cv::KeyPoint &kp2 = m_target->getUndistortedKeypoints()[m_matches[i].dst_idx];
        cv::Mat p3dC1;

        triangulate(kp1, kp2, P1, P2, p3dC1);

        if(!std::isfinite(p3dC1.at<float>(0)) ||
           !std::isfinite(p3dC1.at<float>(1)) ||
           !std::isfinite(p3dC1.at<float>(2))) {
            continue;
        }

        cv::Mat p3dC2 = R*p3dC1+t;

        // Check parallax
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        float cosParallax = normal1.dot(normal2)/(dist1*dist2);
        {
            // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            // TODO ??, modified code
            if(p3dC1.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;

            // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
            if(p3dC2.at<float>(2) <= 0 && cosParallax < 0.99998)
                continue;
        }

        {
            // Check reprojection error in first image
            float im1x, im1y;
            float invZ1 = 1.0/p3dC1.at<float>(2);
            im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
            im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

            float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

            if(squareError1 > threshold)
                continue;

            // Check reprojection error in second image
            float im2x, im2y;
            float invZ2 = 1.0/p3dC2.at<float>(2);
            im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
            im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

            float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

            if(squareError2 > threshold)
                continue;
        }


        vCosParallax.push_back(cosParallax);
        points_3d[m_matches[i].src_idx] = cv::Point3f(p3dC1.at<float>(0), p3dC1.at<float>(1), p3dC1.at<float>(2));
        nGood++;
    }


    if (nGood > 0) {
        sort(vCosParallax.begin(),vCosParallax.end());

        size_t idx = std::min(50, static_cast<int>(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    } else {
        parallax = 0;
    }

    return nGood;
}

bool MonoInitializer::checkFrame(std::shared_ptr<Frame> frame) const
{
    if (frame->getKeypointSize() < MIN_KEYPOINT_NUM) {
        return false;
    } else {
        return true;
    }
}

bool MonoInitializer::checkFrameMatching(const std::vector<Match> &matches)
{
    if (matches.size() < MIN_KEYPOINT_NUM) {
        return false;
    } else {
        return true;
    }
}

std::vector<std::vector<size_t> > constructConstantNumberElementsGroups(int max_index, int number_of_groups,
                                                                        int number_of_group_elements)
{
    // Indices for minimum set selection
    vector<size_t> vAllIndices;
    vAllIndices.reserve(max_index);
    for(int i = 0; i < max_index; i++) {
        vAllIndices.push_back(i);
    }

    vector<size_t> vAvailableIndices;
    // Generate sets of 8 points for each RANSAC iteration
    auto groups = vector< vector<size_t> >(number_of_groups, vector<size_t>(number_of_group_elements, 0));

    DUtils::Random::SeedRandOnce(0);

    for(int it = 0; it < number_of_groups; it++)
    {
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        for(int j = 0; j < number_of_group_elements; j++)
        {
            int randi = DUtils::Random::RandomInt(0, vAvailableIndices.size() - 1);
            int idx = vAvailableIndices[randi];

            groups[it][j] = idx;

            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }
    }

    return groups;
}

void normalizePoints(const std::vector<cv::KeyPoint> &points, std::vector<cv::Point2f> &normalized_points, cv::Mat &T)
{
    const size_t N = points.size();
    normalized_points.resize(N);

    float meanX = 0;
    float meanY = 0;
    for(size_t i = 0; i < N; i++) {
        meanX += points[i].pt.x;
        meanY += points[i].pt.y;
    }
    meanX = meanX / N;
    meanY = meanY / N;

    float meanDevX = 0;
    float meanDevY = 0;
    for(size_t i = 0; i < N; i++) {
        normalized_points[i].x = points[i].pt.x - meanX;
        normalized_points[i].y = points[i].pt.y - meanY;

        meanDevX += fabs(normalized_points[i].x);
        meanDevY += fabs(normalized_points[i].y);
    }
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;
    for(size_t i = 0; i < N; i++) {
        normalized_points[i].x = normalized_points[i].x * sX;
        normalized_points[i].y = normalized_points[i].y * sY;
    }

    T = cv::Mat::eye(3, 3, CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
}

cv::Mat computeH21(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2)
{
    const int N = points1.size();

    cv::Mat A(2*N, 9, CV_32F);

    for(int i=0; i<N; i++) {
        const float u1 = points1[i].x;
        const float v1 = points1[i].y;
        const float u2 = points2[i].x;
        const float v2 = points2[i].y;

        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    return vt.row(8).reshape(0, 3);
}

cv::Mat computeF21(const std::vector<cv::Point2f> &points1, const std::vector<cv::Point2f> &points2)
{
    const int N = points1.size();

    cv::Mat A(N,9,CV_32F);

    for(int i=0; i<N; i++) {
        const float u1 = points1[i].x;
        const float v1 = points1[i].y;
        const float u2 = points2[i].x;
        const float v2 = points2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    cv::Mat u,w,vt;

    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    w.at<float>(2)=0;

    return  u*cv::Mat::diag(w)*vt;
}

void triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    cv::Mat A(4,4,CV_32F);

    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

void decomposeEssentialMatrix(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    cv::SVD::compute(E,w,u,vt);

    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}
