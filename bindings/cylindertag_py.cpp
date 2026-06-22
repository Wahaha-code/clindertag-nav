#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <stdexcept>

#include "CylinderTag.h"

namespace py = pybind11;

namespace {

cv::Mat wrap_array_to_mat(const py::array& array, int expected_channels) {
    if (!py::isinstance<py::array>(array)) {
        throw std::runtime_error("Input must be a numpy array.");
    }

    py::buffer_info buf = array.request();
    if (buf.ndim != 2 && buf.ndim != 3) {
        throw std::runtime_error("Expect HxW or HxWxC image.");
    }

    int rows = static_cast<int>(buf.shape[0]);
    int cols = static_cast<int>(buf.shape[1]);

    if (buf.ndim == 2) {
        if (expected_channels != 1) {
            throw std::runtime_error("Expected a 3-channel image.");
        }
        return cv::Mat(rows, cols, CV_8UC1, buf.ptr, buf.strides[0]);
    }

    int channels = static_cast<int>(buf.shape[2]);
    if (channels != expected_channels) {
        throw std::runtime_error("Unexpected channel count.");
    }

    if (channels == 3) {
        return cv::Mat(rows, cols, CV_8UC3, buf.ptr, buf.strides[0]);
    }
    return cv::Mat();
}

py::dict pose_to_dict(const PoseInfo& pose, const ModelInfo* model) {
    py::dict d;
    d["marker_id"] = pose.markerID;
    if (!pose.rvec.empty()) {
        d["rvec"] = std::vector<double>{
            pose.rvec.at<double>(0, 0),
            pose.rvec.at<double>(1, 0),
            pose.rvec.at<double>(2, 0)
        };
    } else {
        d["rvec"] = std::vector<double>{0.0, 0.0, 0.0};
    }
    if (!pose.tvec.empty()) {
        d["tvec"] = std::vector<double>{
            pose.tvec.at<double>(0, 0),
            pose.tvec.at<double>(1, 0),
            pose.tvec.at<double>(2, 0)
        };
    } else {
    d["tvec"] = std::vector<double>{0.0, 0.0, 0.0};
    }
    d["reprojection_error"] = pose.reprojectionError;
    d["observation_count"] = static_cast<std::size_t>(pose.observationCount);
    if (model != nullptr) {
        d["model_base"] = std::vector<double>{model->base.x, model->base.y, model->base.z};
        d["model_axis"] = std::vector<double>{model->axis.x, model->axis.y, model->axis.z};
    } else {
        d["model_base"] = py::none();
        d["model_axis"] = py::none();
    }
    return d;
}

py::dict marker_to_dict(const MarkerInfo& marker) {
    py::dict d;
    d["marker_id"] = marker.markerID;
    d["feature_pos"] = marker.featurePos;
    d["feature_id_left"] = marker.feature_ID_left;
    d["feature_id_right"] = marker.feature_ID_right;

    py::list corners;
    for (const auto& feature_corners : marker.cornerLists) {
        py::list feature_list;
        for (const auto& pt : feature_corners) {
            feature_list.append(std::vector<double>{pt.x, pt.y});
        }
        corners.append(feature_list);
    }
    d["corner_lists"] = corners;
    return d;
}

}  // namespace

py::array_t<double> mat_to_numpy(const cv::Mat& mat) {
    if (mat.empty()) {
        return py::array_t<double>();
    }
    cv::Mat temp;
    if (mat.type() != CV_64F) {
        mat.convertTo(temp, CV_64F);
    } else {
        temp = mat;
    }
    auto result = py::array_t<double>({temp.rows, temp.cols});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    std::memcpy(ptr, temp.ptr<double>(), sizeof(double) * temp.rows * temp.cols);
    return result;
}

class CylinderTagRunner {
public:
    CylinderTagRunner(const std::string& marker_path,
	                      const std::string& model_path,
	                      const std::string& camera_path,
	                      int adaptive_thresh = 5,
	                      bool enable_subpix = true,
	                      int subpix_dist = 5)
	        : tag_(marker_path),
	          adaptive_thresh_(adaptive_thresh),
	          enable_subpix_(enable_subpix),
	          subpix_dist_(subpix_dist) {
        tag_.loadModel(model_path, model_);
        tag_.loadCamera(camera_path, camera_);
    }

    py::dict process(const py::array_t<uint8_t, py::array::c_style | py::array::forcecast>& image) {
        cv::Mat src = wrap_array_to_mat(image, image.ndim() == 2 ? 1 : 3);

        cv::Mat gray;
        if (src.channels() == 3) {
            cv::cvtColor(src, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = src;
        }

        std::vector<MarkerInfo> markers;
        tag_.detect(gray, markers, adaptive_thresh_, enable_subpix_, subpix_dist_);

        std::vector<PoseInfo> poses;
        tag_.estimatePose(gray, markers, model_, camera_, poses, false);

        py::list marker_list;
        for (const auto& marker : markers) {
            marker_list.append(marker_to_dict(marker));
        }

        py::list pose_list;
        for (const auto& pose : poses) {
            const ModelInfo* model_info = nullptr;
            if (pose.markerID >= 0 && pose.markerID < static_cast<int>(model_.size())) {
                model_info = &model_[static_cast<std::size_t>(pose.markerID)];
            }
            pose_list.append(pose_to_dict(pose, model_info));
        }

        py::dict result;
        result["markers"] = marker_list;
        result["poses"] = pose_list;
        return result;
    }

    py::dict camera_params() const {
        py::dict d;
        d["intrinsic"] = mat_to_numpy(camera_.Intrinsic);
        d["dist_coeffs"] = mat_to_numpy(camera_.distCoeffs);
        return d;
    }

private:
    CylinderTag tag_;
    std::vector<ModelInfo> model_;
    CamInfo camera_;

	int adaptive_thresh_;
	bool enable_subpix_;
	int subpix_dist_;
};

PYBIND11_MODULE(cylindertag_cpp, m) {
    py::class_<CylinderTagRunner>(m, "CylinderTagRunner")
        .def(py::init<const std::string&, const std::string&, const std::string&,
                      int, bool, int>(),
	         py::arg("marker_path"),
	         py::arg("model_path"),
	         py::arg("camera_path"),
	         py::arg("adaptive_thresh") = 5,
	         py::arg("enable_subpix") = true,
	         py::arg("subpix_dist") = 5)
        .def("process", &CylinderTagRunner::process,
             py::arg("image"),
             "Run detection + pose estimation on an RGB/BGR image.")
        .def("camera_params", &CylinderTagRunner::camera_params);
}
