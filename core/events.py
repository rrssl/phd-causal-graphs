class Contact:
    _num_objects = 2

    def __init__(self, first, second, world):
        self.first = first
        self.second = second
        self.world = world

    def __call__(self):
        contact = self.world.contact_test_pair(
            self.first.node(), self.second.node()
        ).get_num_contacts()
        return bool(contact)


class Dummy:
    _num_objects = 0

    def __call__(self):
        return True


class Falling:
    _num_objects = 1

    def __init__(self, body, min_linvel=0):
        self.body = body
        self.min_linvel = abs(min_linvel)

    def __call__(self):
        linvel = self.body.node().get_linear_velocity()[2]
        return linvel < -self.min_linvel


class Inclusion:
    _num_objects = 2

    def __init__(self, inside, outside):
        self.inside = inside
        self.outside = outside

    def __call__(self):
        in_bounds = self.inside.node().get_shape_bounds()
        out_bounds = self.outside.node().get_shape_bounds()
        in_center = in_bounds.get_center() + self.inside.get_pos()
        out_center = out_bounds.get_center() + self.outside.get_pos()
        include = ((in_center - out_center).length()
                   + in_bounds.get_radius()) <= out_bounds.get_radius()
        return include


class NoContact:
    _num_objects = 2

    def __init__(self, first, second, world):
        self.first = first
        self.second = second
        self.world = world

    def __call__(self):
        contact = self.world.contact_test_pair(
            self.first.node(), self.second.node()
        ).get_num_contacts()
        return not contact


class Pivoting:
    _num_objects = 1

    def __init__(self, body, min_angvel=0):
        self.body = body
        self.min_angvel_sq = min_angvel ** 2

    def __call__(self):
        angvel_sq = self.body.node().get_angular_velocity().length_squared()
        return angvel_sq > self.min_angvel_sq


class Rising:
    _num_objects = 1

    def __init__(self, body, min_linvel=0):
        self.body = body
        self.min_linvel = abs(min_linvel)

    def __call__(self):
        linvel = self.body.node().get_linear_velocity()[2]
        return linvel > self.min_linvel


class RollingOn:
    _num_objects = 2

    def __init__(self, rolling, support, world, min_rollang=0):
        self.rolling = rolling
        self.support = support
        self.world = world
        self.min_rollang = min_rollang
        self.start_angle = None

    def __call__(self):
        contact = self.world.contact_test_pair(
            self.rolling.node(), self.support.node()
        ).get_num_contacts()
        if contact:
            if self.start_angle is None:
                self.start_angle = self.rolling.get_quat().get_angle()
                return False
            angle = abs(self.rolling.get_quat().get_angle() - self.start_angle)
            return angle > self.min_rollang
        else:
            self.start_angle = None
            return False


class Stopping:
    _num_objects = 1

    def __init__(self, body, max_linvel=1e-3, max_angvel=1):
        self.body = body
        self.max_linvel_sq = max_linvel ** 2
        self.max_angvel_sq = max_angvel ** 2

    def __call__(self):
        linvel_sq = self.body.node().get_linear_velocity().length_squared()
        angvel_sq = self.body.node().get_angular_velocity().length_squared()
        return (linvel_sq < self.max_linvel_sq
                and angvel_sq < self.max_angvel_sq)


class Toppling:
    _num_objects = 1

    def __init__(self, body, angle):
        self.body = body
        self.angle = angle
        self.start_angle = body.get_r()

    def __call__(self):
        return abs(self.body.get_r() - self.start_angle) >= self.angle + 1
