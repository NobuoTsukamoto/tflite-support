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
//  tensorflow_lite_support/examples/task/vision/desktop:image_classifier_demo \
//  -- \
//  --model_path=/path/to/model.tflite \
//  --image_path=/path/to/image.jpg

#include <iostream>
#include <chrono>

#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "tensorflow_lite_support/cc/port/statusor.h"
#include "tensorflow_lite_support/cc/task/core/external_file_handler.h"
#include "tensorflow_lite_support/cc/task/core/proto/external_file_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/image_classifier.h"
#include "tensorflow_lite_support/cc/task/vision/proto/class_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/classifications_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/proto/image_classifier_options_proto_inc.h"
#include "tensorflow_lite_support/cc/task/vision/utils/frame_buffer_common_utils.h"
#include "tensorflow_lite_support/examples/task/vision/desktop/utils/image_utils.h"

ABSL_FLAG(std::string, model_path, "",
          "Absolute path to the '.tflite' image classifier model.");
ABSL_FLAG(int32, max_results, 5,
          "Maximum number of classification results to display.");
ABSL_FLAG(float, score_threshold, 0,
          "Classification results with a confidence score below this value are "
          "rejected. If >= 0, overrides the score threshold(s) provided in the "
          "TFLite Model Metadata. Ignored otherwise.");
ABSL_FLAG(
    std::vector<std::string>, class_name_whitelist, {},
    "Comma-separated list of class names that acts as a whitelist. If "
    "non-empty, classification results whose 'class_name' is not in this list "
    "are filtered out. Mutually exclusive with 'class_name_blacklist'.");
ABSL_FLAG(
    std::vector<std::string>, class_name_blacklist, {},
    "Comma-separated list of class names that acts as a blacklist. If "
    "non-empty, classification results whose 'class_name' is in this list "
    "are filtered out. Mutually exclusive with 'class_name_whitelist'.");
ABSL_FLAG(int32, num_thread, -1,
          "The number of threads to be used for TFLite ops that support "
          "multi-threading when running inference with CPU."
          "num_threads should be greater than 0 or equal to -1. Setting num_threads to "
          "-1 has the effect to let TFLite runtime set the value.");

namespace tflite {
namespace task {
namespace vision {

// Window name. 
const cv::String kWindowName = "TensorFlow Lite Support Image classification example.";

ImageClassifierOptions BuildOptions() {
  ImageClassifierOptions options;
  options.mutable_model_file_with_metadata()->set_file_name(
      absl::GetFlag(FLAGS_model_path));
  options.set_max_results(absl::GetFlag(FLAGS_max_results));
  options.set_num_threads(absl::GetFlag(FLAGS_num_thread));
  if (absl::GetFlag(FLAGS_score_threshold) >= 0) {
    options.set_score_threshold(absl::GetFlag(FLAGS_score_threshold));
  }
  for (const std::string& class_name :
       absl::GetFlag(FLAGS_class_name_whitelist)) {
    options.add_class_name_whitelist(class_name);
  }
  for (const std::string& class_name :
       absl::GetFlag(FLAGS_class_name_blacklist)) {
    options.add_class_name_blacklist(class_name);
  }
  return options;
}

void DrawCaption(cv::Mat& im,
                 const cv::Point& point,
                 const std::string& caption) {
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 2);
    cv::putText(im, caption, point, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 1);
}

void DisplayResult(const ClassificationResult& result, cv::Mat& image) {
  for (int head = 0; head < result.classifications_size(); ++head) {
    const Classifications& classifications = result.classifications(head);
    for (int rank = 0; rank < classifications.classes_size(); ++rank) {
      const Class& classification = classifications.classes(rank);
      std::ostringstream caption;

      caption << absl::StrFormat("Rank #%d: ", rank);
      if (classification.has_display_name()) {
        caption << classification.display_name();
      } else {
        caption << absl::StrFormat("index: %d", classification.index());
      }
      caption << absl::StrFormat(", score: %.5f", classification.score());

      DrawCaption(image, cv::Point(10, 60 + (rank * 30)), caption.str());
    }
  }
}

absl::Status Classify() {
  // Build ImageClassifier.
  const ImageClassifierOptions& options = BuildOptions();
  ASSIGN_OR_RETURN(std::unique_ptr<ImageClassifier> image_classifier,
                   ImageClassifier::CreateFromOptions(options));

  // OpenCV window setting.
  cv::namedWindow(kWindowName,
      cv::WINDOW_GUI_NORMAL | cv::WINDOW_AUTOSIZE | cv::WINDOW_KEEPRATIO);
  cv::moveWindow(kWindowName, 100, 100);

  // Opencv videocapture setting.
  cv::VideoCapture cap(0);
  auto cap_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
  auto cap_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

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

    // Run classification and display results.
    ASSIGN_OR_RETURN(ClassificationResult result,
                     image_classifier->Classify(*frame_buffer));

    time_span = std::chrono::steady_clock::now() - start_time;
    time_caption << "Inference: " << std::fixed << std::setprecision(2) << time_span.count() << " ms";
    DrawCaption(frame, cv::Point(10, 30), time_caption.str());
    DisplayResult(result, frame);

    // Show image and handle the keyboard before moving to the next frame
    cv::imshow(kWindowName, frame);
    const int key = cv::waitKey(1);
    if (key == 27 || key == 'q') {  // Esc or q key
      break;  // Escape
    }
  }

  // Cleanup and return.
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
  if (!absl::GetFlag(FLAGS_class_name_whitelist).empty() &&
      !absl::GetFlag(FLAGS_class_name_blacklist).empty()) {
    std::cerr << "'class_name_whitelist' and 'class_name_blacklist' arguments "
                 "are mutually exclusive.\n";
    return 1;
  }

  // Run classification.
  absl::Status status = tflite::task::vision::Classify();
  if (status.ok()) {
    return 0;
  } else {
    std::cerr << "Classification failed: " << status.message() << "\n";
    return 1;
  }
}
