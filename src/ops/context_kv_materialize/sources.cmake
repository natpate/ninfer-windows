target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/context_kv_materialize.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/materialize.cu"
  "${CMAKE_CURRENT_LIST_DIR}/context_kv_key_post.cu"
  "${CMAKE_CURRENT_LIST_DIR}/materialize_nvfp4.cu"
)
