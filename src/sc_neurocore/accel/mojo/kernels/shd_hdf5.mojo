# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SC-NeuroCore — Mojo HDF5 recording resource ownership

"""Own HDF5 identifiers and selected native-double variable-length buffers."""

from std.collections import List
from std.ffi import OwnedDLHandle
from std.memory import Pointer


@fieldwise_init
struct Vlen(Copyable, Movable):
    """Match the system HDF5 hvl_t size/pointer layout on a 64-bit host."""

    var count: UInt64
    var data: Int


struct Context(Movable):
    """Own an HDF5 library and its identifiers until explicit checked cleanup.
    """

    var library: OwnedDLHandle
    var objects: List[Tuple[Int64, String]]

    def __init__(out self, path: String) raises:
        """Reserve the bounded identifier stack before loading system HDF5."""
        self.objects = List[Tuple[Int64, String]](capacity=32)
        self.library = OwnedDLHandle(path)
        if self.library.get_function[Int32]("H5open")() < 0:
            raise Error("HDF5 initialisation failed")

    def own(mut self, id: Int64, closer: String) raises -> Int64:
        """Register a successful identifier with its matching HDF5 close function.
        """
        if id < 0:
            raise Error("HDF5 object is unavailable")
        self.objects.append((id, closer))
        return id

    def close(mut self) raises:
        """Close every registered identifier in reverse order and report failure.
        """
        var failed = False
        for i in range(len(self.objects) - 1, -1, -1):
            var entry = self.objects[i]
            if self.library.get_function[Int32](entry[1])(entry[0]) < 0:
                failed = True
        self.objects.clear()
        if failed:
            raise Error("HDF5 identifier close failed")

    def native(self, name: String) raises -> Int64:
        """Resolve an initialised native HDF5 datatype identifier."""
        var symbol = self.library.get_symbol[Int64](name)
        if not symbol:
            raise Error("HDF5 native datatype is unavailable")
        return symbol.value()[]

    def dataset(mut self, file: Int64, var name: String) raises -> Int64:
        """Open a named dataset from the owned read-only recording file."""
        var id = self.library.get_function[Int64]("H5Dopen2")(
            file, name.as_c_string_slice(), Int64(0)
        )
        return self.own(id, "H5Dclose")

    def datatype(mut self, dataset: Int64) raises -> Int64:
        """Own one dataset datatype until checked context cleanup."""
        var id = self.library.get_function[Int64]("H5Dget_type")(dataset)
        return self.own(id, "H5Tclose")

    def selected_space(
        mut self, dataset: Int64, index: Int
    ) raises -> Tuple[Int64, UInt64]:
        """Select a zero-based row of a rank-one dataset and retain its total count.
        """
        var id = self.library.get_function[Int64]("H5Dget_space")(dataset)
        var space = self.own(id, "H5Sclose")
        if (
            self.library.get_function[Int32]("H5Sget_simple_extent_ndims")(
                space
            )
            != 1
        ):
            raise Error("SHD datasets must have rank one")
        var dimensions = UInt64(0)
        var null = Int(0)
        if (
            self.library.get_function[Int32]("H5Sget_simple_extent_dims")(
                space, Pointer(to=dimensions), null
            )
            < 0
        ):
            raise Error("HDF5 extent read failed")
        if UInt64(index) >= dimensions:
            raise Error("SHD recording index is out of range")
        var start = UInt64(index)
        var count = UInt64(1)
        if (
            self.library.get_function[Int32]("H5Sselect_hyperslab")(
                space,
                Int32(0),
                Pointer(to=start),
                null,
                Pointer(to=count),
                null,
            )
            < 0
        ):
            raise Error("HDF5 row selection failed")
        return (space, dimensions)

    def numeric_vlen(mut self, dataset: Int64) raises:
        """Refuse nonnumeric or non-variable-length event vector datatypes."""
        var type = self.datatype(dataset)
        if self.library.get_function[Int32]("H5Tget_class")(type) != 9:
            raise Error("SHD events must be variable-length vectors")
        var id = self.library.get_function[Int64]("H5Tget_super")(type)
        var base = self.own(id, "H5Tclose")
        var kind = self.library.get_function[Int32]("H5Tget_class")(base)
        if kind != 0 and kind != 1:
            raise Error("SHD events must be numeric")

    def vector_bytes(
        self, dataset: Int64, type: Int64, space: Int64
    ) raises -> UInt64:
        """Estimate the selected converted-double allocation before an explicit read.
        """
        var size = UInt64(0)
        if (
            self.library.get_function[Int32]("H5Dvlen_get_buf_size")(
                dataset, type, space, Pointer(to=size)
            )
            < 0
        ):
            raise Error("HDF5 vector allocation estimate failed")
        return size

    def vector(
        self,
        dataset: Int64,
        type: Int64,
        memory: Int64,
        space: Int64,
        maximum: Int,
    ) raises -> List[Float64]:
        """Copy one bounded native-double vector and always reclaim its HDF5 buffer.
        """
        var value = Vlen(UInt64(0), 0)
        try:
            if (
                self.library.get_function[Int32]("H5Dread")(
                    dataset, type, memory, space, Int64(0), Pointer(to=value)
                )
                < 0
            ):
                raise Error("HDF5 vector read failed")
            if value.count > UInt64(maximum // 32):
                raise Error("SHD recording exceeds its event budget")
            if value.count != 0 and value.data == 0:
                raise Error("HDF5 returned an absent vector")
            var result = List[Float64](capacity=Int(value.count))
            if value.count != 0:
                var data = Pointer[Float64, MutAnyOrigin](
                    unsafe_from_address=value.data
                )
                for i in range(Int(value.count)):
                    result.append(data[unsafe_offset=i])
            return result^
        finally:
            if (
                self.library.get_function[Int32]("H5Dvlen_reclaim")(
                    type, memory, Int64(0), Pointer(to=value)
                )
                < 0
            ):
                raise Error("HDF5 vector reclaim failed")

    def label(
        mut self, dataset: Int64, memory: Int64, space: Int64
    ) raises -> Int64:
        """Read signed/unsigned labels and refuse values outside int64."""
        var type = self.datatype(dataset)
        if (
            self.library.get_function[Int32]("H5Tget_class")(type) != 0
            or self.library.get_function[UInt64]("H5Tget_size")(type) > 8
        ):
            raise Error("SHD labels must be representable integers")
        var sign = self.library.get_function[Int32]("H5Tget_sign")(type)
        if sign == 0:
            var value = UInt64(0)
            var native = self.native("H5T_NATIVE_UINT64_g")
            if (
                self.library.get_function[Int32]("H5Dread")(
                    dataset, native, memory, space, Int64(0), Pointer(to=value)
                )
                < 0
            ):
                raise Error("HDF5 label read failed")
            if value > UInt64(0x7FFFFFFFFFFFFFFF):
                raise Error("SHD label exceeds int64")
            return Int64(value)
        if sign != 1:
            raise Error("HDF5 label sign read failed")
        var value = Int64(0)
        var native = self.native("H5T_NATIVE_INT64_g")
        if (
            self.library.get_function[Int32]("H5Dread")(
                dataset, native, memory, space, Int64(0), Pointer(to=value)
            )
            < 0
        ):
            raise Error("HDF5 label read failed")
        return value
