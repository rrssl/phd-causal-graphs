"""
Design space.

"""
import numpy as np


class DesignSpace:
    """Design space of a model.

    Points in this space are obtained by normalizing the free parameters from
    their respective original range to [0, 1] and concatenating them.

    Parameters
    ----------
    ranges : sequence
      Each element is a (o_name, o_ranges) pair, where 'o_ranges' is a sequence
      of pairs of floats (i.e. a range of acceptable values). To fix a value,
      the start and end of the range should be equal.

    """
    def __init__(self, ranges):
        self.names = [name for name, _ in ranges]
        # Compute the array of concatenated origin and scale of ranges. Its
        # length is the same as the vector's. It is used for (de)normalization
        # of the vector.
        self.origin_scale_array = np.array(
            [(a, b-a)
             for _, o_ranges in ranges for a, b in o_ranges if a != b]
        )
        # Compute the (n_objects, n) array of transforms, where fixed values
        # are prefilled. Its length is equal to the number of objects.
        self.xform_array = np.array(
            [a if a == b else np.nan
             for _, o_ranges in ranges for a, b in o_ranges]
        ).reshape(len(ranges), -1)
        # Compute the test array of free parameters.
        self.is_free = np.isnan(self.xform_array)

    def __len__(self):
        return self.origin_scale_array.shape[0]

    def vector2xforms(self, vector):
        """Convert a vector to its transforms dict representation."""
        vector = (np.asarray(vector) * self.origin_scale_array[:, 1]
                  + self.origin_scale_array[:, 0])
        xform_array = self.xform_array.copy()
        xform_array[self.is_free] = vector
        xforms = {name: xform for name, xform in zip(self.names, xform_array)}
        return xforms

    def xforms2vector(self, xforms):
        """Convert a transforms dict to it vector representation."""
        xform_array = np.array([xforms[name] for name in self.names])
        vector = (xform_array[self.is_free] - self.origin_scale_array[:, 0]
                  ) / self.origin_scale_array[:, 1]
        return vector


def load_design_space(scene_data, default_size=6):
    ranges = []
    for obj_data in scene_data:
        name = obj_data['name']
        try:
            o_ranges = obj_data['xform']['range']
        except KeyError:
            o_ranges = None
        try:
            xform = obj_data['xform']['value']
        except KeyError:
            xform = [0] * default_size
        if o_ranges is None:
            o_ranges = [(p, p) for p in xform]
        else:
            o_ranges = [(p, p) if rng is None else tuple(rng)
                        for rng, p in zip(o_ranges, xform)]
        ranges.append((name, o_ranges))
    return DesignSpace(ranges)
