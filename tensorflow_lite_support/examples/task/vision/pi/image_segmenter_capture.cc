/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Example usage:
// bazel run -c opt \
//  tensorflow_lite_support/examples/task/vision/desktop:image_segmenter_capture \
//  -- \
//  --model_path=/path/to/model.tflite

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/core/proto/external_file_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/image_segmenter.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_segmenter_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/segmentations_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

ABSL_FLAG(std::string, model_path, "",
          "Absolute path to the '.tflite' image segmenter model.");
ABSL_FLAG(int32, num_thread, -1,
          "The number of threads to be used for TFLite ops that support "
          "multi-threading when running inference with CPU."
          "num_threads should be greater than 0 or equal to -1. Setting num_threads to "
          "-1 has the effect to let TFLite runtime set the value.");

namespace tflite {
namespace task {
namespace vision {

// Window name. 
const cv::String kWindowName = "TensorFlow Lite Support Segmantation example.";

ImageSegmenterOptions BuildOptions() {
  ImageSegmenterOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      absl::GetFlag(FLAGS_model_path));
  options.set_num_threads(absl::GetFlag(FLAGS_num_thread));
  // Confidence masks are not supported by this tool: output_type is set to
  // CATEGORY_MASK by default.
  return options;
}

std::unique_ptr<cv::Mat> EncodeMaskToMat(const SegmentationResult& result) {
  if (result.segmentation_size() != 1) {
    std::cout << "Image segmentation models with multiple output segmentations are not "
        "supported by this tool." << std::endl;
    return nullptr;
  }
  const Segmentation& segmentation = result.segmentation(0);
  // Extract raw mask data as a uint8 pointer.
  const uint8* raw_mask =
      reinterpret_cast<const uint8*>(segmentation.category_mask().data());

  // Create RgbImageData for the output mask.
  auto seg_im = std::make_unique<cv::Mat>(cv::Size(segmentation.width(), segmentation.height()), CV_8UC3);
  auto wdith = seg_im->cols;
  seg_im->forEach<cv::Vec3b>([&](cv::Vec3b &src, const int position[2]) -> void {
    size_t index = position[0] * wdith + position[1];
    Segmentation::ColoredLabel colored_label =
        segmentation.colored_labels(raw_mask[index]);
        src[0] = colored_label.b();
        src[1] = colored_label.g();
        src[2] = colored_label.r();
    });
  
  return seg_im;
}

void DrawCaption(cv::Mat& im,
                 const cv::Point& point,
                 const std::string& caption) {
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

absl::Status Segment() {
  // Build ImageClassifier.
  const ImageSegmenterOptions& options = BuildOptions();
  ASSIGN_OR_RETURN(std::unique_ptr<ImageSegmenter> image_segmenter,
                   ImageSegmenter::CreateFromOptions(options));

  // OpenCV window setting.
  cv::namedWindow(kWindowName,
      cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
  cv::moveWindow(kWindowName, 100, 100);

  // Opencv videocapture setting.
  cv::VideoCapture cap(0);
  auto cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  auto cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
  auto is_display_label_im = false;

  while(cap.isOpened())
  {
    cv::Mat frame, input_im;
    std::chrono::duration<double, std::milli> time_span;
    std::ostringstream time_caption;

    cap >> frame;
    cv::cvtColor(frame, input_im, cv::COLOR_BGR2RGB);

    const auto& start_time = std::chrono::steady_clock::now();

    // Load image in a FrameBuffer.
    std::unique_ptr<FrameBuffer> frame_buffer;
    frame_buffer = CreateFromRgbRawBuffer(input_im.data, {input_im.cols, input_im.rows});

    // Run segmentation and save category mask.
    ASSIGN_OR_RETURN(SegmentationResult result,
                     image_segmenter->Segment(*frame_buffer));
    auto seg_mat = EncodeMaskToMat(result);

    time_span = std::chrono::steady_clock::now() - start_time;

    cv::resize(*seg_mat, *seg_mat, cv::Size(frame.cols, frame.rows));
    if (!is_display_label_im) {
      frame = (frame / 2) + (*seg_mat / 2);
    }

    time_caption << "Inference: " << std::fixed << std::setprecision(2) << time_span.count() << " ms";
    DrawCaption(frame, cv::Point(10, 30), time_caption.str());

    // Show image and handle the keyboard before moving to the next frame
    cv::imshow(kWindowName, frame);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q') { // Esc or q key
    
      break;  // Escape
    } else if (key == ' ') {
      is_display_label_im = !is_display_label_im;
    }
  }
  return absl::OkStatus();
}

}  // namespace vision
}  // namespace task
}  // namespace tflite

int main(int argc, char** argv) {
  // Parse command line arguments and perform sanity checks.
  absl::ParseCommandLine(argc, argv);
  if (absl::GetFlag(FLAGS_model_path).empty()) {
    std::cerr << "Missing mandatory 'model_path' argument.\n";
    return 1;
  }

  // Run segmentation.
  absl::Status status = tflite::task::vision::Segment();
  if (status.ok()) {
    return 0;
  } else {
    std::cerr << "Segmentation failed: " << status.message() << "\n";
    return 1;
  }
}
