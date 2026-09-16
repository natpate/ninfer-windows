target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/bf16/candidate_selector_path.cu"
  "${CMAKE_CURRENT_LIST_DIR}/bf16/candidate_selector_path_plan.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/nvfp4/candidate_selector_path_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/../wrapper/candidate_selector.cpp"
)
