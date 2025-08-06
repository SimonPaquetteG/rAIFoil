cmake_minimum_required(VERSION 3.12)
project(rAIFoil CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# — Find LibTorch and Boost.Program_options —
find_package(Torch REQUIRED CONFIG)
find_package(Boost 1.65 REQUIRED COMPONENTS program_options)

# — Paths for model embedding —
set(GENERATED_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
file(MAKE_DIRECTORY ${GENERATED_DIR})

# Find xxd utility for binary embedding
find_program(XXD_EXEC xxd REQUIRED
  HINTS /usr/bin /usr/local/bin
  DOC "xxd executable for embedding model binary"
)

# Gather all TorchScript models and PCA data files
file(GLOB MODEL_SRC_LIST "${CMAKE_CURRENT_SOURCE_DIR}/model/architecture/mlp_inference_evalmode_comps*_model*.pt")
file(GLOB PCA_SRC_LIST   "${CMAKE_CURRENT_SOURCE_DIR}/model/pca/pca_transform_*components.dat")

# Lists to collect generated sources, headers, and database entries
set(MODEL_CPP_FILES)
set(MODEL_HEADER_FILES)
set(PCA_CPP_FILES)
set(PCA_HEADER_FILES)
set(ENSEMBLE_ENTRIES)

# Loop over each model file: copy, embed, and generate headers
foreach(model_path IN LISTS MODEL_SRC_LIST)
  get_filename_component(model_name ${model_path} NAME)
  string(REGEX REPLACE "[^A-Za-z0-9]" "_" model_var ${model_name})

  # Copy the model into generated dir
  set(copied_model ${GENERATED_DIR}/${model_name})
  add_custom_command(
    OUTPUT ${copied_model}
    COMMAND ${CMAKE_COMMAND} -E copy ${model_path} ${copied_model}
    DEPENDS ${model_path}
    COMMENT "Copy TorchScript model ${model_name}"
  )

  # Embed the model as C++ source
  set(model_cpp ${GENERATED_DIR}/${model_var}.cpp)
  add_custom_command(
    OUTPUT ${model_cpp}
    COMMAND ${XXD_EXEC} -i ${model_name} ${model_var}.cpp
    DEPENDS ${copied_model}
    WORKING_DIRECTORY ${GENERATED_DIR}
    COMMENT "Embed TorchScript model ${model_name} into C++ source"
  )

  # Generate header for extern declarations
  set(model_header ${GENERATED_DIR}/${model_var}.h)
  file(WRITE ${model_header}
    "// Auto-generated from ${model_name}\n"
    "#pragma once\n"
    "extern unsigned char ${model_var}[];\n"
    "extern unsigned int ${model_var}_len;\n"
  )

  list(APPEND MODEL_CPP_FILES ${model_cpp})
  list(APPEND MODEL_HEADER_FILES ${model_header})

  # Determine matching PCA var and record ensemble entry
  string(REGEX REPLACE "mlp_inference_evalmode_comps([0-9]+)_model[0-9]+\\.pt" "\\1" comp_idx ${model_name})
  string(REGEX REPLACE "[^A-Za-z0-9]" "_" pca_var "pca_transform_${comp_idx}components.dat")
  list(APPEND ENSEMBLE_ENTRIES "{ ${model_var}, ${model_var}_len, ${pca_var}, ${pca_var}_len }")
endforeach()

# Loop over each PCA file: copy, embed, and generate headers
foreach(pca_path IN LISTS PCA_SRC_LIST)
  get_filename_component(pca_name ${pca_path} NAME)
  string(REGEX REPLACE "[^A-Za-z0-9]" "_" pca_var ${pca_name})

  # Copy the PCA data file into generated dir
  set(copied_pca ${GENERATED_DIR}/${pca_name})
  add_custom_command(
    OUTPUT ${copied_pca}
    COMMAND ${CMAKE_COMMAND} -E copy ${pca_path} ${copied_pca}
    DEPENDS ${pca_path}
    COMMENT "Copy PCA parameters ${pca_name}"
  )

  # Embed PCA data as C++ source
  set(pca_cpp ${GENERATED_DIR}/${pca_var}.cpp)
  add_custom_command(
    OUTPUT ${pca_cpp}
    COMMAND ${XXD_EXEC} -i ${pca_name} ${pca_var}.cpp
    DEPENDS ${copied_pca}
    WORKING_DIRECTORY ${GENERATED_DIR}
    COMMENT "Embed PCA parameters ${pca_name} into C++ source"
  )

  # Generate header for extern declarations
  set(pca_header ${GENERATED_DIR}/${pca_var}.h)
  file(WRITE ${pca_header}
    "// Auto-generated from ${pca_name}\n"
    "#pragma once\n"
    "extern unsigned char ${pca_var}[];\n"
    "extern unsigned int ${pca_var}_len;\n"
  )

  list(APPEND PCA_CPP_FILES ${pca_cpp})
  list(APPEND PCA_HEADER_FILES ${pca_header})
endforeach()

# Custom targets to embed all data
add_custom_target(embed_model_data DEPENDS ${MODEL_CPP_FILES})
add_custom_target(embed_pca_data   DEPENDS ${PCA_CPP_FILES})

# Generate ensemble_data.h registry with includes
list(LENGTH ENSEMBLE_ENTRIES ENSEMBLE_SIZE)
set(ensemble_header ${GENERATED_DIR}/ensemble_data.h)
file(WRITE ${ensemble_header}
  "// Auto-generated ensemble data registry\n"
  "#pragma once\n"
  "#include <cstddef>\n"
)
# Include each model and PCA header
foreach(header IN LISTS MODEL_HEADER_FILES PCA_HEADER_FILES)
  file(APPEND ${ensemble_header} "#include \"${header}\"\n")
endforeach()
# Define the ensemble member struct and array
file(APPEND ${ensemble_header}
  "struct EnsembleMember { const unsigned char* model_data; size_t model_len; const unsigned char* pca_data; size_t pca_len; };\n"
  "static const EnsembleMember ensemble_members[] = {\n"
)
foreach(entry IN LISTS ENSEMBLE_ENTRIES)
  file(APPEND ${ensemble_header} "  ${entry},\n")
endforeach()
file(APPEND ${ensemble_header}
  "};\n"
  "static const size_t ENSEMBLE_SIZE = ${ENSEMBLE_SIZE};\n"
)

# — Project source files —
set(SRCS
  utils/airfoil_utils.cpp
  utils/pca_utils.cpp
  utils/scaler_utils.cpp
  utils/help.cpp
  main.cpp
)

# Create executable
add_executable(rAIFoil ${SRCS})

target_include_directories(rAIFoil PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${GENERATED_DIR}
  ${TORCH_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

# Link embedded sources and dependencies
target_sources(rAIFoil PRIVATE ${MODEL_CPP_FILES} ${PCA_CPP_FILES} ${ensemble_header})
add_dependencies(rAIFoil embed_model_data embed_pca_data)

# Link libraries
target_link_libraries(rAIFoil PRIVATE
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Embed RPATH for runtime library discovery
set_target_properties(rAIFoil PROPERTIES
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN/../../local/lib:$ORIGIN/../../local/libtorch/lib"
)
