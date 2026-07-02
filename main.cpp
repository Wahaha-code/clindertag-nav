#include "header/CylinderTag.h"
#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <iomanip>
#include <sstream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

Mat frame, img_gray;
vector<MarkerInfo> markers;
vector<ModelInfo> marker_model;
CamInfo camera;
vector<PoseInfo> pose;

struct DetectionStats {
	std::size_t frames = 0;
	std::size_t frames_with_pose = 0;
};

DetectionStats g_detection_stats;

void read_from_image(const fs::path& image_path,
                     const fs::path& marker_path,
                     const fs::path& model_path,
                     const fs::path& camera_path);
void read_from_video(const fs::path& video_path,
                     const fs::path& marker_path,
                     const fs::path& model_path,
                     const fs::path& camera_path);
void report_pose_quality(const vector<PoseInfo>& pose_infos);
void update_detection_stats(const vector<PoseInfo>& pose_infos);
void print_detection_summary();

int main(int argc, char** argv){
	google::InitGoogleLogging(argv[0]);
	
	fs::path exec_path;
	try {
		exec_path = fs::weakly_canonical(fs::path(argv[0])).parent_path();
	} catch (const fs::filesystem_error&) {
		exec_path = fs::absolute(fs::path(argv[0])).parent_path();
	}
	const fs::path asset_root = exec_path.parent_path();
	const fs::path marker_path = asset_root / "CTag_2f12c.marker";
	const fs::path model_path = asset_root / "CTag_2f12c_d32.model";
	const fs::path camera_path = asset_root / "cameraParams.yml";

	fs::path input_path = (argc > 1)
		? fs::absolute(fs::path(argv[1]))
		: asset_root / "test.avi";

	if (!fs::exists(marker_path) || !fs::exists(model_path) || !fs::exists(camera_path)) {
		cerr << "Required data files not found. Expected them under: "
		     << asset_root << endl;
		return EXIT_FAILURE;
	}

	if (!fs::exists(input_path)) {
		cerr << "Input file not found: " << input_path << endl;
		return EXIT_FAILURE;
	}

	string ext = input_path.extension().string();
	transform(ext.begin(), ext.end(), ext.begin(),
	          [](unsigned char ch){ return static_cast<char>(tolower(ch)); });
	static const vector<string> image_ext = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"};
	const bool is_image = any_of(image_ext.begin(), image_ext.end(),
		[&](const string& candidate){ return ext == candidate; });

	g_detection_stats = DetectionStats{};

	if (is_image) {
		read_from_image(input_path, marker_path, model_path, camera_path);
	} else {
		read_from_video(input_path, marker_path, model_path, camera_path);
	}

	print_detection_summary();
	waitKey(0);
	destroyAllWindows();

	return 0;
}

void read_from_image(const fs::path& image_path,
                     const fs::path& marker_path,
                     const fs::path& model_path,
                     const fs::path& camera_path){
	frame = imread(image_path.string());
	if (frame.empty()) {
		cerr << "Failed to load image: " << image_path << endl;
		return;
	}

	CylinderTag marker(marker_path.string());
	marker.loadModel(model_path.string(), marker_model);
	marker.loadCamera(camera_path.string(), camera);

	if (frame.channels() == 3) {
		cvtColor(frame, img_gray, COLOR_BGR2GRAY);
	} else {
		img_gray = frame.clone();
	}

	marker.detect(img_gray, markers, 5, true, 5);
	marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
	report_pose_quality(pose);
	update_detection_stats(pose);
	marker.drawAxis(img_gray, markers, marker_model, pose, camera, 30);
}

void read_from_video(const fs::path& video_path,
                     const fs::path& marker_path,
                     const fs::path& model_path,
                     const fs::path& camera_path){
	VideoCapture capture(video_path.string());
	if (!capture.isOpened()) {
		cerr << "Failed to open video: " << video_path << endl;
		return;
	}

	CylinderTag marker(marker_path.string());
	marker.loadModel(model_path.string(), marker_model);
	marker.loadCamera(camera_path.string(), camera);

	while (capture.read(frame))
	{	
		if (frame.channels() == 3) {
			cvtColor(frame, img_gray, COLOR_BGR2GRAY);
		} else {
			img_gray = frame.clone();
		}
		markers.clear();
		pose.clear();
		marker.detect(img_gray, markers, 5, true, 5);
		marker.estimatePose(img_gray, markers, marker_model, camera, pose, false);
		report_pose_quality(pose);
		const bool has_pose = !pose.empty();
		update_detection_stats(pose);
		marker.drawAxis(img_gray, markers, marker_model, pose, camera, 30, has_pose ? 0 : 1);
	}
}

void report_pose_quality(const vector<PoseInfo>& pose_infos){
	for (const auto& info : pose_infos) {
		if (info.markerID < 0 || info.observationCount == 0 || info.reprojectionError < 0.0) {
			continue;
		}
		std::ostringstream oss;
		oss << std::fixed << std::setprecision(3) << info.reprojectionError;
		cout << "[PoseQuality] marker " << info.markerID
		     << " reprojection RMS = " << oss.str()
		     << " px (" << info.observationCount << " pts)" << endl;
	}
}

void update_detection_stats(const vector<PoseInfo>& pose_infos){
	++g_detection_stats.frames;
	if (!pose_infos.empty()) {
		++g_detection_stats.frames_with_pose;
	}
}

void print_detection_summary(){
	if (g_detection_stats.frames == 0) {
		cout << "[DetectionSummary] No frames processed." << endl;
		return;
	}
	const double rate = static_cast<double>(g_detection_stats.frames_with_pose) /
	                    static_cast<double>(g_detection_stats.frames);
	cout << "[DetectionSummary] Frames=" << g_detection_stats.frames
	     << ", Successful PnP=" << g_detection_stats.frames_with_pose
	     << ", Success Rate=" << fixed << setprecision(3) << (rate * 100.0)
	     << "%" << endl;
}
