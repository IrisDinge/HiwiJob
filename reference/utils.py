import os
import itertools
import time
import datetime
import json
import random
import numpy
import PIL.Image
import queue
import concurrent.futures
import traceback
import ctypes
import torch
import torch.nn as nn
import tensorflow

hannover_path = "/home/gritzner/tmp/data/hannover"
vaihingen = {"path": "/data/geoTL/daten/IPI/Vaihingen/top", "gt_path": "/home/gritzner/tmp/data/vaihingen",
             "images": ("01", "03", "05", "07", "11", "13", "15", "17", "21", "23", "26", "28", "30", "32", "34", "37")}

get_batch_lib = ctypes.CDLL("/home/gritzner/Dokumente/Projects/DeepRS/Prototype/c/libget_batch.so")
get_batch_uint8 = get_batch_lib.get_batch_uint8
get_batch_uint8.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32),
                            numpy.ctypeslib.ndpointer(ctypes.c_uint8), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
                            numpy.ctypeslib.ndpointer(ctypes.c_double),
                            numpy.ctypeslib.ndpointer(ctypes.c_int64)]
get_batch_uint8.restype = None
get_batch_ext_uint8 = get_batch_lib.get_batch_ext_uint8
get_batch_ext_uint8.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32),
                                numpy.ctypeslib.ndpointer(ctypes.c_uint8), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
                                numpy.ctypeslib.ndpointer(ctypes.c_double), numpy.ctypeslib.ndpointer(ctypes.c_double),
                                numpy.ctypeslib.ndpointer(ctypes.c_int64)]
get_batch_ext_uint8.restype = None
get_batch_float32 = get_batch_lib.get_batch_float32
get_batch_float32.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32),
                              numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
                              numpy.ctypeslib.ndpointer(ctypes.c_double),
                              numpy.ctypeslib.ndpointer(ctypes.c_int64)]
get_batch_float32.restype = None
get_batch_ext_float32 = get_batch_lib.get_batch_ext_float32
get_batch_ext_float32.argtypes = [numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_int32),
                                  numpy.ctypeslib.ndpointer(ctypes.c_float), numpy.ctypeslib.ndpointer(ctypes.c_uint8),
                                  numpy.ctypeslib.ndpointer(ctypes.c_double),
                                  numpy.ctypeslib.ndpointer(ctypes.c_double),
                                  numpy.ctypeslib.ndpointer(ctypes.c_int64)]
get_batch_ext_float32.restype = None


def load_image(path, resize_factor=0, use_lanczos=True, as_numpy=True):
    image = PIL.Image.open(path)
    if resize_factor > 0:
        new_size = (int(image.size[0] * resize_factor), int(image.size[1] * resize_factor))
        image = image.resize(new_size, PIL.Image.LANCZOS if use_lanczos else PIL.Image.NEAREST)
    if as_numpy:
        image = numpy.asarray(image, dtype=numpy.float32 if image.mode == "F" else numpy.uint8)
        if len(image.shape) > 2:
            assert len(image.shape) == 3
            image = channels_last2first(image)
        else:
            image = numpy.expand_dims(image, axis=0)
            assert len(image.shape) == 3 and image.shape[0] == 1
        if not image.flags["C_CONTIGUOUS"]:
            image = numpy.ascontiguousarray(image)
    return image


def channels_last2first(image):
    return numpy.transpose(image, (2, 0, 1))


def channels_first2last(image):
    return numpy.transpose(image, (1, 2, 0))


def load_hannover_irgb(resize_factor=0, skip_eilenriede=False):
    images = []
    ground_truth = []
    for i, j in itertools.product(range(4), range(4)):
        if (i + 4 * j) == 3 and skip_eilenriede:
            continue
        path = os.path.join(hannover_path, f"ir{i}{j}.png")
        image = load_image(path, resize_factor=resize_factor)
        path = os.path.join(hannover_path, f"rgb{i}{j}.png")
        image = numpy.concatenate((image, load_image(path, resize_factor=resize_factor)), axis=0)
        if not image.flags["C_CONTIGUOUS"]:
            image = numpy.ascontiguousarray(image)
        images.append(image)
        path = os.path.join(hannover_path, f"gt{i}{j}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth


def load_hannover_irgbd(resize_factor=0, skip_eilenriede=False):
    images = []
    ground_truth = []
    for i, j in itertools.product(range(4), range(4)):
        if (i + 4 * j) == 3 and skip_eilenriede:
            continue
        path = os.path.join(hannover_path, f"ir{i}{j}.png")
        image = load_image(path, resize_factor=resize_factor)
        path = os.path.join(hannover_path, f"rgb{i}{j}.png")
        image = numpy.concatenate((image, load_image(path, resize_factor=resize_factor)), axis=0)
        image = numpy.asarray(image, dtype=numpy.float32) / 255.0
        path = os.path.join(hannover_path, f"dom{i}{j}.tif")
        image = numpy.concatenate((image, load_image(path, resize_factor=resize_factor)), axis=0)
        if not image.flags["C_CONTIGUOUS"]:
            image = numpy.ascontiguousarray(image)
        images.append(image)
        path = os.path.join(hannover_path, f"gt{i}{j}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth


def load_vaihingen_irg(resize_factor=0):
    images = []
    ground_truth = []
    for i in vaihingen["images"]:
        path = os.path.join(vaihingen["path"], f"top_mosaic_09cm_area{i}.tif")
        images.append(load_image(path, resize_factor=resize_factor))
        path = os.path.join(vaihingen["gt_path"], f"top_mosaic_09cm_area{i}.png")
        ground_truth.append(load_image(path, resize_factor=resize_factor, use_lanczos=False))
    return images, ground_truth


def draw_bootstrap_sets(sample_range, num_sets=10, samples_per_set=10):
    bootstrap = {}
    for i in range(num_sets):
        training_set = []
        for j in range(samples_per_set):
            training_set.append(random.randrange(sample_range))
        training_set = tuple(sorted(training_set))
        test_set = tuple([x for x in range(sample_range) if not x in training_set])
        bootstrap[i] = (training_set, test_set)
    return bootstrap


def load_bootstrap_sets(path):
    bootstrap = {}
    with open(path, "r") as f:
        temp = json.load(f)
        for key, values in temp.items():
            training_set = tuple([int(x) for x in values[0]])
            test_set = tuple([int(x) for x in values[1]])
            bootstrap[int(key)] = (training_set, test_set)
    return bootstrap


def get_transform(h_flip=False, v_flip=False, x_shear=0, y_shear=0, rotation=0, tx=0, ty=0):
    transform = numpy.eye(3, dtype=numpy.float64)
    # TODO: this is a very hacky way of implementing flipping and should probably not be used
    if h_flip:  # horizontal flipping
        transform[0, 2] = 1
    if v_flip:  # vertical flipping
        transform[1, 2] = 1
    if x_shear != 0:  # horizontal shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 1] = numpy.tan(x_shear * numpy.pi / 180)
        transform = numpy.matmul(temp, transform)
    if y_shear != 0:  # vertical shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[1, 0] = numpy.tan(y_shear * numpy.pi / 180)
        transform = numpy.matmul(temp, transform)
    if rotation != 0:  # rotation
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 2] = -0.5
        temp[1, 2] = -0.5
        transform = numpy.matmul(temp, transform)
        alpha = rotation * numpy.pi / 180
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 0] = numpy.cos(alpha)
        temp[1, 0] = numpy.sin(alpha)
        temp[0, 1] = -temp[1, 0]
        temp[1, 1] = temp[0, 0]
        transform = numpy.matmul(temp, transform)
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 2] = 0.5
        temp[1, 2] = 0.5
        transform = numpy.matmul(temp, transform)
    if tx != 0 or ty != 0:  # translation ("cropping")
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 2] = tx
        temp[1, 2] = ty
        transform = numpy.matmul(temp, transform)
    if not transform.flags["C_CONTIGUOUS"]:
        transform = numpy.ascontiguousarray(transform)
    return transform


def get_random_transform(h_flip=False, v_flip=False, x_shear_range=0, y_shear_range=0, rotation_range=0, tx_range=0,
                         ty_range=0):
    # best for Hannover (17.04.2019): no flipping, shearing range 16, rotation range 45, translation range 1
    params = {}
    params["h_flip"] = h_flip and random.random() < 0.5
    params["v_flip"] = v_flip and random.random() < 0.5
    params["x_shear"] = random.uniform(-x_shear_range, x_shear_range)
    params["y_shear"] = random.uniform(-y_shear_range, y_shear_range)
    params["rotation"] = random.uniform(-rotation_range, rotation_range)
    params["tx"] = random.uniform(0, tx_range)
    params["ty"] = random.uniform(0, ty_range)
    return get_transform(**params)


def get_nested_transform(x_shear=0, y_shear=0, rotation=0, tx=0, ty=0, scaling=0):
    transform = numpy.eye(3, dtype=numpy.float64)
    if x_shear != 0:  # horizontal shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 1] = numpy.tan(x_shear * numpy.pi / 180)
        transform = numpy.matmul(temp, transform)
    if y_shear != 0:  # vertical shearing
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[1, 0] = numpy.tan(y_shear * numpy.pi / 180)
        transform = numpy.matmul(temp, transform)
    if rotation != 0:  # rotation
        alpha = rotation * numpy.pi / 180
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 0] = numpy.cos(alpha)
        temp[1, 0] = numpy.sin(alpha)
        temp[0, 1] = -temp[1, 0]
        temp[1, 1] = temp[0, 0]
        transform = numpy.matmul(temp, transform)
    if tx != 0 or ty != 0:  # translation ("cropping")
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 2] = tx
        temp[1, 2] = ty
        transform = numpy.matmul(temp, transform)
    if scaling != 0:  # scaling
        if scaling > 0:
            scaling += 1
        else:
            scaling = 1 / (1 - scaling)
        temp = numpy.eye(3, dtype=numpy.float64)
        temp[0, 0] = scaling
        temp[1, 1] = scaling
        transform = numpy.matmul(temp, transform)
    if not transform.flags["C_CONTIGUOUS"]:
        transform = numpy.ascontiguousarray(transform)
    return transform


def get_random_nested_transform(x_shear_range=0, y_shear_range=0, rotation_range=0, tx_range=0, ty_range=0,
                                scaling_range=0):
    params = {}
    params["x_shear"] = random.uniform(-x_shear_range, x_shear_range)
    params["y_shear"] = random.uniform(-y_shear_range, y_shear_range)
    params["rotation"] = random.uniform(-rotation_range, rotation_range)
    params["tx"] = random.uniform(-tx_range, tx_range)
    params["ty"] = random.uniform(-ty_range, ty_range)
    params["scaling"] = random.uniform(-scaling_range, scaling_range);
    return get_nested_transform(**params)


def get_mini_batch(mini_batch_size, patch_size, image, ground_truth, num_classes, transforms, nested_transforms=None):
    assert image.shape[1] == ground_truth.shape[1]
    assert image.shape[2] == ground_truth.shape[2]
    assert transforms.shape[0] == mini_batch_size
    assert nested_transforms is None or nested_transforms.shape[0] == mini_batch_size
    shape_info = (mini_batch_size, patch_size, patch_size, image.shape[1], image.shape[2], image.shape[0], num_classes)
    shape_info = numpy.ascontiguousarray(shape_info, dtype=numpy.int64)
    mini_batch = numpy.empty((mini_batch_size, image.shape[0], patch_size, patch_size), dtype=numpy.float32)
    if not mini_batch.flags["C_CONTIGUOUS"]:
        mini_batch = numpy.ascontiguousarray(mini_batch)
    mini_batch_gt = numpy.zeros((mini_batch_size, patch_size, patch_size), dtype=numpy.int32)
    if not mini_batch_gt.flags["C_CONTIGUOUS"]:
        mini_batch_gt = numpy.ascontiguousarray(mini_batch_gt)
    if nested_transforms is None:
        get_batch = get_batch_uint8 if image.dtype == numpy.uint8 else get_batch_float32
        get_batch(mini_batch, mini_batch_gt, image, ground_truth, transforms, shape_info)
    else:
        get_batch = get_batch_ext_uint8 if image.dtype == numpy.uint8 else get_batch_ext_float32
        get_batch(mini_batch, mini_batch_gt, image, ground_truth, transforms, nested_transforms, shape_info)
    return mini_batch, mini_batch_gt


def get_validation_data(images, ground_truth, subset, patch_size, num_classes):
    num_patches = 0
    for index in subset:
        image = images[index]
        hor_patches = ((image.shape[1] - 1) // patch_size) + 1
        ver_patches = ((image.shape[2] - 1) // patch_size) + 1
        num_patches += (hor_patches * ver_patches)
    val_images = numpy.empty((num_patches, images[0].shape[0], patch_size, patch_size), dtype=numpy.float32)
    val_gt = numpy.empty((num_patches, patch_size, patch_size), dtype=numpy.int32)
    offset = 0
    scale_factor = (1.0 / 255.0) if images[
                                        0].dtype == numpy.uint8 else 1.0  # assumes that either all images are of type uint8 or all images are of type float32
    for index in subset:
        image = images[index]
        gt = ground_truth[index]
        hor_patches = ((image.shape[1] - 1) // patch_size) + 1
        ver_patches = ((image.shape[2] - 1) // patch_size) + 1
        for x, y in itertools.product(range(hor_patches), range(ver_patches)):
            left = x * patch_size if (x + 1) * patch_size < image.shape[1] else image.shape[1] - patch_size
            top = y * patch_size if (y + 1) * patch_size < image.shape[2] else image.shape[2] - patch_size
            val_images[offset, :, :, :] = image[:, left:left + patch_size, top:top + patch_size] * scale_factor
            val_gt[offset, :, :] = gt[0, left:left + patch_size, top:top + patch_size]
            offset += 1
    return val_images, val_gt


def get_num_slices(data, slice_size):
    assert data.shape[0] > 0
    return ((data.shape[0] - 1) // slice_size) + 1


def get_slice(data, slice_index, slice_size):
    if isinstance(data, tuple) or isinstance(data, list):
        l = len(data)
        for i in range(1, l):
            assert data[0].shape[0] == data[i].shape[0]
        return [get_slice(data[i], slice_index, slice_size) for i in range(l)]
    else:
        begin = slice_index * slice_size
        end = (slice_index + 1) * slice_size
        if end > data.shape[0]:
            end = data.shape[0]
        index = [slice(begin, end)]
        for i in range(1, len(data.shape)):
            index.append(slice(0, data.shape[i]))
        return data.__getitem__(tuple(index))


def slice_generator(data, slice_size):
    if isinstance(data, tuple) or isinstance(data, list):
        l = len(data)
        for i in range(1, l):
            assert data[0].shape[0] == data[i].shape[0]
        num_slices = get_num_slices(data[0], slice_size)
        for i in range(num_slices):
            yield [get_slice(data[j], i, slice_size) for j in range(l)]
    else:
        num_slices = get_num_slices(data, slice_size)
        for i in range(num_slices):
            yield get_slice(data, i, slice_size)


class ModelFitter:
    def __init__(self, num_epochs, num_mini_batches, shuffle=True, output_path=None, history_filename="history.json",
                 max_queue_size=4):
        self.num_epochs = num_epochs
        self.num_mini_batches = num_mini_batches
        self.shuffle = shuffle
        self.output_path = output_path
        self.history_filename = history_filename
        self.max_queue_size = (max_queue_size - 1) if max_queue_size > 1 else 1

    def initialize(self):
        pass

    def pre_epoch(self, epoch):
        pass

    def get_batch(self, epoch, batch, batch_data):
        pass

    def train(self, epoch, batch, batch_data, metrics):
        pass

    def post_epoch(self, epoch, metrics):
        pass

    def finalize(self):
        pass

    def fit(self):
        self.history = {}
        self.initialize()
        for epoch in range(self.num_epochs):
            epoch_timestamp = time.perf_counter()
            print(f"starting epoch {epoch + 1} at", datetime.datetime.now().time().strftime("%H:%M:%S"))
            self.pre_epoch(epoch)
            batch_ids = numpy.arange(self.num_mini_batches)
            if self.shuffle:
                numpy.random.shuffle(batch_ids)
            self._fit_epoch(epoch, batch_ids)
            metrics = {key: numpy.mean(self.history[key][-self.num_mini_batches:]) for key in self._metrics_keys}
            self.post_epoch(epoch, metrics)
            for key in self._metrics_keys:
                del metrics[key]
            for key, value in metrics.items():
                if not key in self.history:
                    self.history[key] = []
                self.history[key].append(value)
            if self.output_path != None and self.history_filename != None:
                filename = f"{self.output_path}/{self.history_filename}"
                print(f"saving history to '{filename}'...")
                with open(filename, "w") as f:
                    json.dump(self.history, f)
            epoch_timestamp = time.perf_counter() - epoch_timestamp
            self._progress(epoch, -1, epoch_timestamp, metrics)
        self.finalize()

    def _fit_epoch(self, epoch, batch_ids):
        self._last_line_length = None
        self._metrics_keys = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            try:
                q = queue.Queue(maxsize=self.max_queue_size)
                executor.submit(self._get_batches, epoch, batch_ids, q)
                for iteration, batch in enumerate(batch_ids):
                    batch_timestamp = time.perf_counter()
                    metrics = {}
                    self.train(epoch, batch, q.get(), metrics)
                    batch_timestamp = time.perf_counter() - batch_timestamp
                    for key, value in metrics.items():
                        if not key in self._metrics_keys:
                            self._metrics_keys.add(key)
                        if not key in self.history:
                            self.history[key] = []
                        values = self.history[key]
                        values.append(value)
                        metrics[key] = numpy.mean(values[-(iteration + 1):])
                    self._progress(epoch, iteration, batch_timestamp, metrics)
            except:
                traceback.print_exc()
        print("")

    def _get_batches(self, epoch, batch_ids, q):
        try:
            for batch in batch_ids:
                data = []
                self.get_batch(epoch, batch, data)
                q.put(data)
        except:
            traceback.print_exc()

    def _progress(self, epoch, iteration, elapsed_time, metrics):
        s = f"epoch = {epoch + 1}/{self.num_epochs}"
        if iteration >= 0:
            s = f"{s}, iteration = {iteration + 1}/{self.num_mini_batches}"
        if elapsed_time < 1:
            elapsed_time *= 1000
            unit = "ms"
            if elapsed_time < 1:
                elapsed_time *= 1000
                unit = "us"
            elapsed_time = round(elapsed_time)
            s = f"{s}, time = {elapsed_time}{unit}"
        elif elapsed_time < 60:
            s = f"{s}, time = {elapsed_time:.2f}s"
        else:
            elapsed_time /= 60
            unit = "min"
            if elapsed_time >= 60:
                elapsed_time /= 60
                unit = "h"
                if elapsed_time >= 24:
                    elapsed_time /= 24
                    unit = "d"
            s = f"{s}, time = {elapsed_time:.1f}{unit}"
        for key, value in metrics.items():
            s = f"{s}, {key} = {value}"
        if self._last_line_length and iteration >= 0 and len(s) < self._last_line_length:
            s += (self._last_line_length - len(s)) * " "
        print(s, end="\r" if iteration >= 0 else "\n\n")
        self._last_line_length = len(s)


def predict(net, data, mini_batch_size, as_float=True):
    results = []
    device = next(net.parameters()).device
    with torch.no_grad():
        for x in slice_generator(data, mini_batch_size):
            x = torch.from_numpy(x)
            if as_float:
                x = x.float()
            x = x.to(device)
            results.append(net(x).cpu())
    return torch.cat(results, 0).numpy()


def reduce_iterations_to_epochs(data, num_mini_batches, reduce_fn=numpy.mean):
    assert len(data) % num_mini_batches == 0
    num_epochs = len(data) // num_mini_batches
    return [reduce_fn(data[epoch * num_mini_batches:(epoch + 1) * num_mini_batches]) for epoch in range(num_epochs)]


def load_deep_lab_v3p_weights(net, input_map=None):
    def collect_layers(net):
        result = []
        for layer in net:
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d):
                result.append(layer)
            elif isinstance(layer, nn.ModuleList) or isinstance(layer, nn.Sequential):
                result.extend(collect_layers(layer))
        return result

    layers = collect_layers(net.part1)
    layers.extend(collect_layers(net.part2))

    mapping = {
        "MobilenetV2/Conv": 0,
        "MobilenetV2/Conv/BatchNorm": 1,
        "MobilenetV2/expanded_conv/depthwise": 4,
        "MobilenetV2/expanded_conv/depthwise/BatchNorm": 5,
        "MobilenetV2/expanded_conv/project": 6,
        "MobilenetV2/expanded_conv/project/BatchNorm": 7
    }

    for index in range(1, 17):
        target = len(mapping) + 2
        mapping[f"MobilenetV2/expanded_conv_{index}/expand"] = target
        mapping[f"MobilenetV2/expanded_conv_{index}/expand/BatchNorm"] = target + 1
        mapping[f"MobilenetV2/expanded_conv_{index}/depthwise"] = target + 2
        mapping[f"MobilenetV2/expanded_conv_{index}/depthwise/BatchNorm"] = target + 3
        mapping[f"MobilenetV2/expanded_conv_{index}/project"] = target + 4
        mapping[f"MobilenetV2/expanded_conv_{index}/project/BatchNorm"] = target + 5

    conv_shape_map = {2: 1, 3: 0, 0: 2, 1: 3}
    depthwise_conv_shape_map = {2: 0, 3: 1, 0: 2, 1: 3}

    # download URL for pre-trained weights: https://github.com/qixuxiang/deeplabv3plus/blob/master/model/deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz
    # expected MD5: 4020e71ab31636648101b000cca3b6b4
    # expected SHA-256: 2b7fe43461c2d9d56b3ed6baed5547fc066361ba12ab84c8e85f0c31914c034f
    reader = tensorflow.compat.v1.train.NewCheckpointReader(
        "/home/gritzner/tmp/models/DeepRS/deeplabv3_mnv2_pascal_trainval/model.ckpt-30000")

    for key, target in mapping.items():
        layer = layers[target]
        if "BatchNorm" in key:
            for torch_name, tf_name in (
            ("weight", "gamma"), ("bias", "beta"), ("running_mean", "moving_mean"), ("running_var", "moving_variance")):
                pre_trained_tensor = reader.get_tensor(f"{key}/{tf_name}")
                tensor = layer._parameters.get(torch_name, None)
                tensor = layer._buffers.get(torch_name, tensor)
                if pre_trained_tensor.shape[0] != tensor.shape[0]:
                    raise RuntimeError(
                        f"tensor shape mismatch for {key}/{tf_name}: {pre_trained_tensor.shape} vs. {tensor.shape}")
                tensor = torch.from_numpy(pre_trained_tensor).float().to(tensor.device)
                layer.__setattr__(torch_name, tensor if "running" in torch_name else nn.Parameter(tensor))
        else:
            weights_string = "weights" if layer.groups == 1 else "depthwise_weights"
            shape_map = conv_shape_map if layer.groups == 1 else depthwise_conv_shape_map
            pre_trained_tensor = reader.get_tensor(f"{key}/{weights_string}")
            tensor = layer.weight
            # assume square kernels
            assert pre_trained_tensor.shape[0] == pre_trained_tensor.shape[1]
            assert tensor.shape[2] == tensor.shape[3]
            for i, j in shape_map.items():
                if pre_trained_tensor.shape[i] != tensor.shape[j]:
                    raise RuntimeError(
                        f"tensor shape mismatch for {key}/{weights_string}: {pre_trained_tensor.shape} vs. {tensor.shape}")
            weights = numpy.empty(tensor.shape)
            for i, j in itertools.product(range(pre_trained_tensor.shape[2]), range(pre_trained_tensor.shape[3])):
                s = slice(0, tensor.shape[3])
                index = [s, s, s, s]
                index[shape_map[2]] = i if target != 0 or not input_map else input_map[i]
                index[shape_map[3]] = j
                weights.__setitem__(tuple(index), pre_trained_tensor.__getitem__((s, s, i, j)))
            layer.weight = nn.Parameter(torch.from_numpy(weights).float().to(tensor.device))
