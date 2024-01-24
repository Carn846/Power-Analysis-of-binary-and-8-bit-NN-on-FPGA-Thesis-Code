# Generalised Synthesis code used for the findBestFolding Script. Only works on the MNIST datasets. CIFAR-10 didn't get to this stage.
# Will need to ensure that the paths to the networks are set correctly.

import random


class Test:
    def __init__(self) -> None:
        self.hiddenNumber = random.randint(1,1000)
        print(f'The hidden number is {self.hiddenNumber}')
    def check(self,value):
        # Lowest it can go is a folding of 50
        assert value >= self.hiddenNumber
        return 1

class NetworkSynthesis:
    def networkSynthesisMNIST(self,foldingValue,networktype):
        # Load network
        # Currently split up into different types to enable the system to run sequentially and minimise computer down time
        import torch
        if networktype == 1:
            from finn.util.test import get_test_model_trained
            Network = get_test_model_trained("TFC",1, 1)
            inputVector = (1,1,28,28)
            network_name = f"TFC_folding_amount{foldingValue}" # Allow for easy adaptability of the network
        if networktype == 2:
            use_finn = True
            if use_finn:
                from finn.util.test import get_test_model_trained
                Network = get_test_model_trained("TFC",2, 2)
                network_name = f"FINN_Trinary_folding_amount{foldingValue}" # Allow for easy adaptability of the network
            else:
                from MNIST import QuantTrinaryFCMNIST
                Network = QuantTrinaryFCMNIST()
                PATH = f'/home/julien/finn/notebooks/Thesis/Training_log/MNIST NETWORKS2023-08-20/MNISTv2-2bit.pth' # Set correctly
                Network.load_state_dict(torch.load(PATH)) 
                network_name = f"QuantTrinaryFCMNIST_folding_amount{foldingValue}" # Allow for easy adaptability of the network
            inputVector = (1,1,28,28)
            
        if networktype == 8:
            # the 8 bit one
            from MNIST import FC8bit
            Network = FC8bit()
            inputVector = (1,1,28,28)
            network_name = f"FC8bitv4_folding_amount{foldingValue}" # Allow for easy adaptability of the network
            PATH = f'/home/julien/finn/notebooks/Thesis/Training_log/MNIST NETWORKS2023-08-20/FC8bit.pth' # Set Correctly
            Network.load_state_dict(torch.load(PATH))
        
        # Convert to Qonnx
        from finn.util.basic import make_build_dir
        from finn.util.visualization import showInNetron
        import os

        import onnx
        from finn.util.test import get_test_model_trained
        import brevitas.onnx as bo
        from qonnx.core.modelwrapper import ModelWrapper
        from qonnx.transformation.infer_shapes import InferShapes
        from qonnx.transformation.fold_constants import FoldConstants
        from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
            
        build_dir = os.environ["FINN_BUILD_DIR"]
        build_dir = f'{os.getcwd()}/TestSynth'
        # Saving the network - taken from the demo
        bo.export_finn_onnx(Network, inputVector, build_dir + f"/{network_name}_export.onnx")
        model = ModelWrapper(build_dir + f"/{network_name}_export.onnx")
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(RemoveStaticGraphInputs())
        model.save(build_dir + f"/{network_name}_tidy.onnx")
        # Just loading all the used modules
        from finn.util.pytorch import ToTensor
        from qonnx.transformation.merge_onnx_models import MergeONNXModels
        from qonnx.core.datatype import DataType
        from qonnx.transformation.insert_topk import InsertTopK
        from qonnx.transformation.infer_datatypes import InferDataTypes
        # Although not doing preprocessing will still do the model calls so that if you do use it you can just insert
        # your instructions here.
        model = ModelWrapper(build_dir+f"/{network_name}_tidy.onnx")
        # add input quantization annotation: UINT8 for all BNN-PYNQ models
        global_inp_name = model.graph.input[0].name
        model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
        # postprocessing: insert Top-1 node at the end
        model = model.transform(InsertTopK(k=1))

        # Tidy up network
        model = model.transform(InferShapes())
        model = model.transform(FoldConstants())
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(GiveReadableTensorNames())
        model = model.transform(InferDataTypes())
        model = model.transform(RemoveStaticGraphInputs())
        model.save(build_dir+f"/{network_name}_pre_post.onnx")

        # Streamlining
        from finn.transformation.streamline import Streamline
        import finn
        from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
        from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
        import finn.transformation.streamline.absorb as absorb
        from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
        from qonnx.transformation.infer_data_layouts import InferDataLayouts
        from qonnx.transformation.general import RemoveUnusedTensors

        model = ModelWrapper(build_dir + f"/{network_name}_pre_post.onnx")
        model = model.transform(MoveScalarLinearPastInvariants())
        model = model.transform(Streamline())
        model = model.transform(LowerConvsToMatMul())
        model = model.transform(MakeMaxPoolNHWC())
        model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
        model = model.transform(ConvertBipolarMatMulToXnorPopcount())
        model = model.transform(Streamline())
        # absorb final add-mul nodes into TopK
        model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
        model = model.transform(InferDataLayouts())
        model = model.transform(RemoveUnusedTensors())
        model.save(build_dir+f"/{network_name}_streamlined.onnx")


        # Convert to HLS equivalent
        import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
        from finn.transformation.fpgadataflow.create_dataflow_partition import (
            CreateDataflowPartition,
        )
        import finn.builder.build_dataflow 
        from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
        from qonnx.custom_op.registry import getCustomOp
        from qonnx.transformation.infer_data_layouts import InferDataLayouts

        # choose the memory mode for the MVTU units, decoupled or const
        mem_mode = "decoupled" # smaller memory foot print. Longer synth times use 'const'

        model = ModelWrapper(build_dir + f"/{network_name}_streamlined.onnx")
        model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))

        model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
        # TopK to LabelSelect
        model = model.transform(to_hls.InferLabelSelectLayer())
        # input quantization (if any) to standalone thresholding
        model = model.transform(to_hls.InferThresholdingLayer())
        model = model.transform(to_hls.InferConvInpGen())
        model = model.transform(to_hls.InferStreamingMaxPool())
        # get rid of Reshape(-1, 1) operation between hlslib nodes
        #model = model.transform(RemoveCNVtoFCFlatten()) # comment out when not using any conv layers
        # get rid of Tranpose -> Tranpose identity seq
        model = model.transform(absorb.AbsorbConsecutiveTransposes())
        # infer tensor data layouts
        model = model.transform(InferDataLayouts())

        # Partitioning the network
        parent_model = model.transform(CreateDataflowPartition())
        parent_model.save(build_dir + f"/{network_name}_dataflow_parent.onnx")
        sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        #print(sdp_node)
        sdp_node = getCustomOp(sdp_node)
        dataflow_model_filename = sdp_node.get_nodeattr("model")
        # save the dataflow partition with a different name for easier access
        dataflow_model = ModelWrapper(dataflow_model_filename)
        dataflow_model.save(build_dir + f"/{network_name}_dataflow_model.onnx")

        # automatic setting of folding
        import finn.transformation.fpgadataflow.set_folding as SetFolding
        import finn.transformation.fpgadataflow.set_fifo_depths as InsertFIFO
        from finn.util.basic import pynq_part_map
        fpga = "Pynq-Z2"
        fpgapart = pynq_part_map[fpga]
        model = ModelWrapper(build_dir + f"/{network_name}_dataflow_model.onnx")
        #model = model.transform(InsertFIFO.RemoveShallowFIFOs())
        model = model.transform(InsertFIFO.InsertFIFO()) # this seems to generate actual hls layers. I am currently of the opinion that these layers actually don't work with the system?
        model = model.transform(SetFolding.SetFolding(target_cycles_per_frame=foldingValue))
        model.save(build_dir + f"/{network_name}_folded.onnx")

        # Run synthesis of network
        from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
        import finn.transformation.fpgadataflow.prepare_ip as prepare_ip
        import finn.transformation.fpgadataflow.insert_iodma as insert_iodma
        pynq_board = "Pynq-Z2"
        target_clk_ns = 10

        model = ModelWrapper(build_dir+f"/{network_name}_folded.onnx")
        #the best effort generator. Doesn't seem to want to work for some reason.
        #model = model.transform(insert_iodma.InsertIODMA()) # Handled by below code
        model = model.transform(ZynqBuild(platform = pynq_board, period_ns = target_clk_ns)) # handles all the synthesis parts
        #model = model.transform(prepare_ip.PrepareIP(fpgapart=fpgapart,clk=target_clk_ns)) #create for custom blocks. Not needed here
        model.save(build_dir+f"/{network_name}_synthesised.onnx")

        # Resources used
        model = ModelWrapper(build_dir+f"/{network_name}_synthesised.onnx")
        sdp_node_middle = getCustomOp(model.graph.node[1])
        postsynth_layers = sdp_node_middle.get_nodeattr("model")
        model = ModelWrapper(build_dir+f"/{network_name}_synthesised.onnx")
        from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
        model = model.transform(AnnotateResources('synth'))
        model.save(build_dir+f"/{network_name}_synthesised_resources.onnx")

        # Make the pynz zip file
        from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
        model = model.transform(MakePYNQDriver("zynq-iodma"))
        model.save(build_dir + f"/{network_name}_synth.onnx")

        from shutil import copy
        from distutils.dir_util import copy_tree

        # create directory for deployment files
        deployment_dir = make_build_dir(prefix="pynq_deployment_")
        model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

        # get and copy necessary files
        # .bit and .hwh file
        bitfile = model.get_metadata_prop("bitfile")
        hwh_file = model.get_metadata_prop("hw_handoff")
        deploy_files = [bitfile, hwh_file]

        for dfile in deploy_files:
            if dfile is not None:
                copy(dfile, deployment_dir)

        # driver.py and python libraries
        pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
        copy_tree(pynq_driver_dir, deployment_dir)

        from shutil import make_archive
        make_archive(f'{network_name}', 'zip', deployment_dir)
        print(f"done {network_name}")

    def networkSynthesisCIFAR10(self,foldingValue,networktype):
            # TODO:
                # Will only work on this if I have the time

            # Load network
            import torch
            if networktype == 1:
                from finn.util.test import get_test_model_trained
                Network = get_test_model_trained("TFC",1, 1)
                inputVector = (1,1,28,28)
                network_name = f"TFC_folding_amount{foldingValue}" # Allow for easy adaptability of the network
            if networktype == 2:
                use_finn = True
                if use_finn:
                    from finn.util.test import get_test_model_trained
                    Network = get_test_model_trained("TFC",2, 2)
                    network_name = f"FINN_Trinary_folding_amount{foldingValue}" # Allow for easy adaptability of the network
                else:
                    from MNIST import QuantTrinaryFCMNIST
                    Network = QuantTrinaryFCMNIST()
                    PATH = f'/home/julien/finn/notebooks/Thesis/Training_log/MNIST NETWORKS2023-08-20/MNISTv2-2bit.pth'
                    Network.load_state_dict(torch.load(PATH)) 
                    network_name = f"QuantTrinaryFCMNIST_folding_amount{foldingValue}" # Allow for easy adaptability of the network
                inputVector = (1,1,28,28)
                
            if networktype == 8:
                # the 8 bit one
                from MNIST import FC8bit
                Network = FC8bit()
                inputVector = (1,1,28,28)
                network_name = f"FC8bitv4_folding_amount{foldingValue}" # Allow for easy adaptability of the network
                PATH = f'/home/julien/finn/notebooks/Thesis/Training_log/MNIST NETWORKS2023-08-20/FC8bit.pth'
                Network.load_state_dict(torch.load(PATH))
            
            # Convert to Qonnx
            from finn.util.basic import make_build_dir
            from finn.util.visualization import showInNetron
            import os

            import onnx
            from finn.util.test import get_test_model_trained
            import brevitas.onnx as bo
            from qonnx.core.modelwrapper import ModelWrapper
            from qonnx.transformation.infer_shapes import InferShapes
            from qonnx.transformation.fold_constants import FoldConstants
            from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs
                
            build_dir = os.environ["FINN_BUILD_DIR"]
            build_dir = f'{os.getcwd()}/TestSynth'
            # Saving the network - taken from the demo
            bo.export_finn_onnx(Network, inputVector, build_dir + f"/{network_name}_export.onnx")
            model = ModelWrapper(build_dir + f"/{network_name}_export.onnx")
            model = model.transform(InferShapes())
            model = model.transform(FoldConstants())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(RemoveStaticGraphInputs())
            model.save(build_dir + f"/{network_name}_tidy.onnx")
            # Just loading all the used modules
            from finn.util.pytorch import ToTensor
            from qonnx.transformation.merge_onnx_models import MergeONNXModels
            from qonnx.core.datatype import DataType
            from qonnx.transformation.insert_topk import InsertTopK
            from qonnx.transformation.infer_datatypes import InferDataTypes
            # Although not doing preprocessing will still do the model calls so that if you do use it you can just insert
            # your instructions here.
            model = ModelWrapper(build_dir+f"/{network_name}_tidy.onnx")
            # add input quantization annotation: UINT8 for all BNN-PYNQ models
            global_inp_name = model.graph.input[0].name
            model.set_tensor_datatype(global_inp_name, DataType["UINT8"])
            # postprocessing: insert Top-1 node at the end
            model = model.transform(InsertTopK(k=1))

            # Tidy up network
            model = model.transform(InferShapes())
            model = model.transform(FoldConstants())
            model = model.transform(GiveUniqueNodeNames())
            model = model.transform(GiveReadableTensorNames())
            model = model.transform(InferDataTypes())
            model = model.transform(RemoveStaticGraphInputs())
            model.save(build_dir+f"/{network_name}_pre_post.onnx")

            # Streamlining
            from finn.transformation.streamline import Streamline
            import finn
            from qonnx.transformation.lower_convs_to_matmul import LowerConvsToMatMul
            from qonnx.transformation.bipolar_to_xnor import ConvertBipolarMatMulToXnorPopcount
            import finn.transformation.streamline.absorb as absorb
            from finn.transformation.streamline.reorder import MakeMaxPoolNHWC, MoveScalarLinearPastInvariants
            from qonnx.transformation.infer_data_layouts import InferDataLayouts
            from qonnx.transformation.general import RemoveUnusedTensors

            model = ModelWrapper(build_dir + f"/{network_name}_pre_post.onnx")
            model = model.transform(MoveScalarLinearPastInvariants())
            model = model.transform(Streamline())
            model = model.transform(LowerConvsToMatMul())
            model = model.transform(MakeMaxPoolNHWC())
            model = model.transform(absorb.AbsorbTransposeIntoMultiThreshold())
            model = model.transform(ConvertBipolarMatMulToXnorPopcount())
            model = model.transform(Streamline())
            # absorb final add-mul nodes into TopK
            model = model.transform(absorb.AbsorbScalarMulAddIntoTopK())
            model = model.transform(InferDataLayouts())
            model = model.transform(RemoveUnusedTensors())
            model.save(build_dir+f"/{network_name}_streamlined.onnx")


            # Convert to HLS equivalent
            import finn.transformation.fpgadataflow.convert_to_hls_layers as to_hls
            from finn.transformation.fpgadataflow.create_dataflow_partition import (
                CreateDataflowPartition,
            )
            import finn.builder.build_dataflow 
            from finn.transformation.move_reshape import RemoveCNVtoFCFlatten
            from qonnx.custom_op.registry import getCustomOp
            from qonnx.transformation.infer_data_layouts import InferDataLayouts

            # choose the memory mode for the MVTU units, decoupled or const
            mem_mode = "decoupled" # smaller memory foot print. Longer synth times use 'const'

            model = ModelWrapper(build_dir + f"/{network_name}_streamlined.onnx")
            model = model.transform(to_hls.InferBinaryMatrixVectorActivation(mem_mode))

            model = model.transform(to_hls.InferQuantizedMatrixVectorActivation(mem_mode))
            # TopK to LabelSelect
            model = model.transform(to_hls.InferLabelSelectLayer())
            # input quantization (if any) to standalone thresholding
            model = model.transform(to_hls.InferThresholdingLayer())
            model = model.transform(to_hls.InferConvInpGen())
            model = model.transform(to_hls.InferStreamingMaxPool())
            # get rid of Reshape(-1, 1) operation between hlslib nodes
            #model = model.transform(RemoveCNVtoFCFlatten()) # comment out when not using any conv layers
            # get rid of Tranpose -> Tranpose identity seq
            model = model.transform(absorb.AbsorbConsecutiveTransposes())
            # infer tensor data layouts
            model = model.transform(InferDataLayouts())

            # Partitioning the network
            parent_model = model.transform(CreateDataflowPartition())
            parent_model.save(build_dir + f"/{network_name}_dataflow_parent.onnx")
            sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
            #print(sdp_node)
            sdp_node = getCustomOp(sdp_node)
            dataflow_model_filename = sdp_node.get_nodeattr("model")
            # save the dataflow partition with a different name for easier access
            dataflow_model = ModelWrapper(dataflow_model_filename)
            dataflow_model.save(build_dir + f"/{network_name}_dataflow_model.onnx")

            # automatic setting of folding
            import finn.transformation.fpgadataflow.set_folding as SetFolding
            import finn.transformation.fpgadataflow.set_fifo_depths as InsertFIFO
            from finn.util.basic import pynq_part_map
            fpga = "Pynq-Z2"
            fpgapart = pynq_part_map[fpga]
            model = ModelWrapper(build_dir + f"/{network_name}_dataflow_model.onnx")
            #model = model.transform(InsertFIFO.RemoveShallowFIFOs())
            model = model.transform(InsertFIFO.InsertFIFO()) # this seems to generate actual hls layers. I am currently of the opinion that these layers actually don't work with the system?
            model = model.transform(SetFolding.SetFolding(target_cycles_per_frame=foldingValue))
            model.save(build_dir + f"/{network_name}_folded.onnx")

            # Run synthesis of network
            from finn.transformation.fpgadataflow.make_zynq_proj import ZynqBuild
            import finn.transformation.fpgadataflow.prepare_ip as prepare_ip
            import finn.transformation.fpgadataflow.insert_iodma as insert_iodma
            pynq_board = "Pynq-Z2"
            target_clk_ns = 10

            model = ModelWrapper(build_dir+f"/{network_name}_folded.onnx")
            #the best effort generator. Doesn't seem to want to work for some reason.
            #model = model.transform(insert_iodma.InsertIODMA()) # Handled by below code
            model = model.transform(ZynqBuild(platform = pynq_board, period_ns = target_clk_ns)) # handles all the synthesis parts
            #model = model.transform(prepare_ip.PrepareIP(fpgapart=fpgapart,clk=target_clk_ns)) #create for custom blocks. Not needed here
            model.save(build_dir+f"/{network_name}_synthesised.onnx")

            # Resources used
            model = ModelWrapper(build_dir+f"/{network_name}_synthesised.onnx")
            sdp_node_middle = getCustomOp(model.graph.node[1])
            postsynth_layers = sdp_node_middle.get_nodeattr("model")
            model = ModelWrapper(build_dir+f"/{network_name}_synthesised.onnx")
            from finn.transformation.fpgadataflow.annotate_resources import AnnotateResources
            model = model.transform(AnnotateResources('synth'))
            model.save(build_dir+f"/{network_name}_synthesised_resources.onnx")

            # Make the pynz zip file
            from finn.transformation.fpgadataflow.make_pynq_driver import MakePYNQDriver
            model = model.transform(MakePYNQDriver("zynq-iodma"))
            model.save(build_dir + f"/{network_name}_synth.onnx")

            from shutil import copy
            from distutils.dir_util import copy_tree

            # create directory for deployment files
            deployment_dir = make_build_dir(prefix="pynq_deployment_")
            model.set_metadata_prop("pynq_deployment_dir", deployment_dir)

            # get and copy necessary files
            # .bit and .hwh file
            bitfile = model.get_metadata_prop("bitfile")
            hwh_file = model.get_metadata_prop("hw_handoff")
            deploy_files = [bitfile, hwh_file]

            for dfile in deploy_files:
                if dfile is not None:
                    copy(dfile, deployment_dir)

            # driver.py and python libraries
            pynq_driver_dir = model.get_metadata_prop("pynq_driver_dir")
            copy_tree(pynq_driver_dir, deployment_dir)

            from shutil import make_archive
            make_archive(f'{network_name}', 'zip', deployment_dir)
            print(f"done {network_name}")
