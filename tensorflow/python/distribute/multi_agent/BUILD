load("//tensorflow:strict.default.bzl", "py_strict_library", "py_strict_test")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

py_strict_library(
    name = "task",
    srcs = ["task.py"],
)

py_strict_library(
    name = "agent",
    srcs = ["agent.py"],
    deps = [":task"],
)

py_strict_library(
    name = "coordinator",
    srcs = ["coordinator.py"],
    deps = [
        ":agent",
        ":task",
    ],
)

py_strict_library(
    name = "multi_agent",
    srcs = ["__init__.py"],
    deps = [
        ":agent",
        ":coordinator",
        ":task",
    ],
)

py_strict_test(
    name = "coordinator_test",
    size = "small",
    srcs = ["coordinator_test.py"],
    deps = [
        ":agent",
        ":coordinator",
        ":task",
        "//tensorflow/python/platform:client_testlib",
    ],
)
