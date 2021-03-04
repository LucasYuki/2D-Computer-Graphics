local facade  = require"facade"
local image   = require"image"
local chronos = require"chronos"
local filter  = require"filter"

local unpack = table.unpack
local floor  = math.floor
local ceil  = math.ceil

local _M = facade.driver()
setmetatable(_ENV, { __index = _M } )

local background = _M.solid_color(_M.color.white)

local error_lim = 0.01

------------------------------------
--        Print functions         --
------------------------------------

local function stderr(...)
    io.stderr:write(string.format(...))
end

local function print_mat(matrix) --display matrix
    for i=1, #matrix do
        temp = ""
        for j=1, #matrix[1] do
            temp = string.format("%s%f ", temp, matrix[i][j])
        end
        print(temp)
    end
end

local function print_vec(vector) --display vector
    temp = ""
    for i=1, #vector do
        temp = string.format("%s%f ", temp, vector[i])
    end
    print(temp)
end

local print_shape = {
    [shape_type.triangle] = function(shape)
        local tdata = shape:get_triangle_data()
        print("\tp1", tdata:get_x1(), tdata:get_y1())
        print("\tp2", tdata:get_x2(), tdata:get_y2())
        print("\tp3", tdata:get_x3(), tdata:get_y3())
    end,

    [shape_type.circle] = function(shape)
        local cdata = shape:get_circle_data()
        print("\tc", cdata:get_cx(), cdata:get_cy())
        print("\tr", cdata:get_r())
    end,

    [shape_type.polygon] = function(shape)
        local pdata = shape:get_polygon_data()
        local coords = pdata:get_coordinates()
        for i = 2, #coords, 2 do
            local xi, yi = coords[i-1], coords[i]
            print("", i//2, xi, yi)
        end
    end,

    [shape_type.rect] = function(shape)
        local rdata = shape:get_rect_data()
        print("", rdata:get_x(), rdata:get_y())
        print("", rdata:get_width(), rdata:get_height())
    end,

    [shape_type.path] = function(shape)
        local pdata = shape:get_path_data()
        pdata:iterate(
            filter.make_input_path_f_find_monotonic_parameters(
                filter.make_input_path_f_find_cubic_parameters({
            root_dx_parameter = function(self, t)
                print("", "root_dx_parameter", t)
            end,
            root_dy_parameter = function(self, t)
                print("", "root_dy_parameter", t)
            end,
            inflection_parameter = function(self, t)
                print("", "inflection_parameter", t)
            end,
            double_point_parameter = function(self, t)
                print("", "double_point_parameter", t)
            end,
            begin_contour = function(self, x0, y0)
                print("", "begin_contour", x0, y0)
            end,
            end_open_contour = function(self, x0, y0)
                print("", "end_open_contour", x0, y0)
            end,
            end_closed_contour = function(self, x0, y0)
                print("", "end_closed_contour", x0, y0)
            end,
            linear_segment = function(self, x0, y0, x1, y1)
                print("", "linear_segment", x0, y0, x1, y1)
            end,
            quadratic_segment = function(self, x0, y0, x1, y1, x2, y2)
                print("", "quadratic_segment", x0, y0, x1, y1, x2, y2)
            end,
            cubic_segment = function(self, x0, y0, x1, y1, x2, y2, x3, y3)
                print("", "cubic_segment", x0, y0, x1, y1, x2, y2, x3, y3)
            end,
            rational_quadratic_segment = function(self, x0, y0, x1, y1, w1,
                                                  x2, y2)
                print("", "rational_quadratic_segment", x0, y0, x1, y1, w1,
                      x2, y2)
            end,
        })))
    end,
}

local function print_color_ramp(ramp)
    print("", ramp:get_spread())
    for i, stop in ipairs(ramp:get_color_stops()) do
        print("", stop:get_offset(), "->", table.unpack(stop:get_color()))
    end
end

local print_paint = {
    [paint_type.solid_color] = function(paint)
        local color = paint:get_solid_color()
        print("", table.unpack(color))
        local opacity = paint:get_opacity()
        print("", "opacity", opacity)
    end,

    [paint_type.linear_gradient] = function(paint)
        local lg = paint:get_linear_gradient_data()
        print("", "p1", lg:get_x1(), lg:get_y1())
        print("", "p2", lg:get_x2(), lg:get_y2())
        print_color_ramp(lg:get_color_ramp())
        local opacity = paint:get_opacity()
        print("", "opacity", opacity)
    end,

    [paint_type.radial_gradient] = function(paint)
        local lg = paint:get_radial_gradient_data()
        print("", "c", lg:get_cx(), lg:get_cy())
        print("", "f", lg:get_fx(), lg:get_fy())
        print("", "r", lg:get_r())
        print_color_ramp(lg:get_color_ramp())
        local opacity = paint:get_opacity()
        print("", "opacity", opacity)
    end
}

------------------------------------
--        Math functions          --
------------------------------------

local function copy_matrix(t)
    local temp = {}
    for i=1, #t do
        temp[i] = {}
        for j=1, #t[1] do
            temp[i][j] = t[i][j]
        end
    end
    return temp
end

local function transpose(matrix)
    local temp = {}
    for i = 1, #matrix[1] do
        temp[i] = {}
        for j = 1, #matrix do
            temp[i][j] = matrix[j][i]
        end
    end
    return temp
end

local function matmul(a, b) --matrix multiplication
    dim_a = {#a, #a[1]}
    dim_b = {#b, #b[1]}
    if dim_a[2] ~= dim_b[1] then
        print(string.format("a: [%d, %d] \nb: [%d, %d]", dim_a[1], dim_a[2], dim_b[1], dim_b[2]))
        error("matrix size incompatible.")
    end
    result = {}
    for i=1, dim_a[1] do
        result[i] = {}
        for j=1, dim_b[2] do
            temp = 0
            for k=1, dim_a[2] do
                temp = temp + a[i][k] * b[k][j]
            end
            result[i][j] = temp
        end
    end
    return result
end

local function transform(matrix, x, y) --affine transform
    local p = {x, y, 1}
    local r = {0, 0}
    for i=1, 3 do
        r[1] = r[1]+p[i]*matrix[1][i]
        r[2] = r[2]+p[i]*matrix[2][i]
    end
    return r[1], r[2]
end

local function gauss_jordan(matrix, s) --Gaussian elimination
    function swap(x, r1, r2) --swap rows r1 and r2 in the matrix x or elements r1 and r2 in the vector x
        temp = x[r1]
        x[r1] = x[r2]
        x[r2] = temp
        return x
    end
    n_row = #matrix
    n_col = #matrix[1]
    local MAX = min(n_row, n_col)
    pivot = 1
    for j=1, MAX do
        if matrix[pivot][j]==0 then --search for the next pivot
            for t=j+1, n_row do
                if matrix[t][j]~=0 then
                    matrix = swap(matrix, t, j)
                    s = swap(s, t, j)
                    break
                end
            end
        end
        if matrix[pivot][j]~=0 then
            if matrix[pivot][j]~=1 then --if the pivot isn't 1, scale the line to turn intro 1
                for n=1, #s[1] do
                    s[pivot][n] = s[pivot][n]/matrix[pivot][j]
                end
                for n=j+1, n_col do
                    matrix[pivot][n] = matrix[pivot][n]/matrix[pivot][j]
                end
                matrix[pivot][j] = 1
            end
            for i=1, n_row do
                if matrix[i][j]~=0 and i~=pivot then
                    for n=1, #s[1] do
                        s[i][n] = s[i][n]-matrix[i][j]*s[pivot][n]
                    end
                    for n=j+1, n_col do
                        matrix[i][n] = matrix[i][n]-matrix[i][j]*matrix[pivot][n]
                    end
                    matrix[i][j] = 0
                end
            end
            pivot = pivot + 1
        end
    end
    return matrix, s
end

local function get_t_vec(t, n)
    local result = {{1}}
    for p=2, n+1 do
        result[1][p] = result[1][p-1]*t
    end
    return result
end

local function b_search(index, k) --binary search
    if k>=index[#index] then
        return #index+1
    end
    local t_min = 1
    local t_max = #index
    local t = 0
    while t_min<t_max do
        t = floor((t_min+t_max)/2)
        if index[t]>k then
            t_max = t
        else
            t_min = t+1
        end
    end
    return t_min
end

local change_of_basis = {
    [3]={{ 1, 0, 0},
         {-2, 2, 0},
         { 1,-2, 1}},
    [4]={{ 1, 0, 0, 0},
         {-3, 3, 0, 0},
         { 3,-6, 3, 0},
         {-1, 3,-3, 1}}
}

local function get_plane(a, b) --cross product
    return {a[2]*b[3]-a[3]*b[2],
            a[3]*b[1]-a[1]*b[3],
            a[1]*b[2]-a[2]*b[1]}
end

local function planes_intersection(p1, p2)
    local matrix = copy_matrix({p1, p2, {0, 0, 1}})
    local _, p = gauss_jordan(matrix, {{0}, {0}, {1}})
    p = transpose(p)[1]
    return p
end

local function is_colinear(p1, p2, p3)
    local v1 = {}
    local v2 = {}
    for i=1, 3 do
        v1[i] = p2[i]-p1[i]
        v2[i] = p3[i]-p1[i]
    end
    local prod = get_plane(v1, v2) --cross product
    if prod[1]==0 and prod[2]==0 and prod[3]==0 then
        return true
    else
        return false
    end
end

local derivative_mat = {
    [3] = {{0, 1, 0},
           {0, 0, 2},
           {0, 0, 0}},
    [4] = {{0, 1, 0, 0},
           {0, 0, 2, 0},
           {0, 0, 0, 3},
           {0, 0, 0, 0}}
}

local function derivate(C)
    return matmul(derivative_mat[#C], C)
end

local function det3(M)
    local result = 0
    for i=0, 2 do
        result = result + M[i+1][1]*M[(i+1)%3+1][2]*M[(i+2)%3+1][3] - M[i+1][3]*M[(i+1)%3+1][2]*M[(i+2)%3+1][1]
    end
    return result
end

local function project(p)
    if not p[3] then
        return p
    end
    return {p[1]/p[3], p[2]/p[3]}
end

local function cubicBspline(nr)
    local k, l, m = 1/(6*nr), 1/(2*nr), 1/nr
    local function calc(b)
        x = {b, b^2, b^3}
        return {x[3]*k,
                (-x[3]+x[2]+x[1]+1/3)*l,
                (x[3]/2-x[2]+2/3)*m,
                ((-x[3]+1)/3+x[2]-x[1])*l}
    end
    return calc
end

------------------------------------
--            Classes             --
------------------------------------

local TransformStack = {}
TransformStack.__index = TransformStack

setmetatable(TransformStack, {
  __call = function (cls, ...)
    return cls.create(...)
  end,
})

function TransformStack.create()
    local self = setmetatable({}, TransformStack)
    self.values = {[0]=_M.identity():toxform()}
    self.len = 0
    return self
end

function TransformStack:push(xf)
    xf = self.values[self.len]*xf
    self.len = self.len + 1
    self.values[self.len] = xf
end

function TransformStack:pop()
    if self.len == 0 then
        error("stack is already empty")
    end
    temp = self.values[self.len]
    self.values[self.len] = nil
    self.len = self.len - 1
    return temp
end

function TransformStack:top()
    if self.len < 0 then
        error("stack is empty")
    end
    return self.values[self.len]
end

local PixelColor= {}
PixelColor.__index = PixelColor

setmetatable(PixelColor, {
  __call = function (cls, ...)
    return cls.create(...)
  end,
})

function PixelColor.create()
    local self = setmetatable({}, PixelColor)
    self.rgb = {0, 0, 0}
    self.alpha = 0
    return self
end

function PixelColor:blend(bottom_color)
    local i_alpha = 1-self.alpha
    for c=1, 3 do
        self.rgb[c] = self.rgb[c]+i_alpha*bottom_color[c]
    end
    self.alpha = self.alpha + i_alpha*bottom_color[4]
    if self.alpha>0.95 then
        return true
    else
        return false
    end
end

local gamma = 2.2
local igamma = 1/gamma
function PixelColor:unpack()
    return self.rgb[1]^gamma, self.rgb[2]^gamma, self.rgb[3]^gamma, self.alpha^gamma
end

function apply_gamma(c, w)
    return (c[1]/w)^igamma, (c[2]/w)^igamma, (c[3]/w)^igamma, (c[4]/w)^igamma
end

local winding_buffer= {}
winding_buffer.__index = winding_buffer

setmetatable(winding_buffer, {
  __call = function (cls, ...)
    return cls.create(...)
  end,
})

function winding_buffer.create()
    local self = setmetatable({}, winding_buffer)
    self.buff = {}
    self.sorted = false
    return self
end

function winding_buffer:add(tbuff)
    self.sorted = false
    for i, values in pairs(tbuff) do
        if self.buff[i] == nil then
            self.buff[i] = {values}
        else
            table.insert(self.buff[i], values)
        end
    end
end

function winding_buffer:get()
    if true then --not self.sorted then
        self.sorted = true
        for i, values in pairs(self.buff) do
            table.sort(values, function(a, b) return a[1]<b[1] end)
        end
    end
    return self.buff
end

------------------------------------
--             Paths              --
------------------------------------

--## Verify if the monotone is increasing or decreasing
local function get_monotone_type(p1, p2)
    local dy = 0
    if p1[2]==p2[2] then
        return nil
    elseif p1[1]<p2[1] then
        dy = p2[2]-p1[2]
    else
        dy = p1[2]-p2[2]
    end
    if dy>0 then
        return 1
    else
        return -1
    end
end

--## Get the min/max values of x and y in a list of points
local function get_box_lim(p)
    local x_lim = {}
    local y_lim = {}
    for i=1, #p do
        local temp = project(p[i])
        x_lim[i] = temp[1]
        y_lim[i] = temp[2]
    end
    table.sort(x_lim)
    table.sort(y_lim)
    local x_inf = x_lim[1]
    local x_sup = x_lim[#x_lim]
    local y_inf = y_lim[1]
    local y_sup = y_lim[#y_lim]
    return x_inf, x_sup, y_inf, y_sup
end

--## Generate a function to verify if the point is on the left of the line ##--
local function left_line(rp1, rp2, rt)
    local p1 = project(rp1)
    local p2 = project(rp2)
    local a = p2[2]-p1[2]
    local mon_type = get_monotone_type(p1, p2)

    local x_min, x_max, y_min, y_max = get_box_lim({p1, p2})

    local signal = 1
    local left = {}
    if a~=0 then --if the line isn't horizontal
        local b = -(p1[1]-p2[1])/a
        local c = p1[1]-p1[2]*b
        if a<0 then
            signal = -1
        end
        if rt then
            local t = project(rt)
            if t[1]<=b*t[2]+c then
                return true
            else
                return false
            end
        end
        left = function (x, y)
            if y<y_min or y>=y_max then return 0
            elseif x<x_min then return signal
            elseif x>=x_max then return 0
            elseif x<=b*y+c then return signal
            else return 0 end
        end
    else
        left = function(x, y) --if the line is horizontal
            return 0
        end
    end
    return left, {x_min, x_max, y_min, y_max, {mon_type, signal}}
end

--## Generate a function that calculate the value at the point ##--
local calc_implicit = {
    [2] = function (iC)
        return function (x, y)
            local temp = matmul({{x, y, 1}}, iC)[1]
            return temp[1]^2-(temp[2]*temp[3])
        end
    end,
    [3] = function (iC)
        if #iC == 3 then
            return function (x, y)
                local temp = matmul({{x, y, 1}}, iC)[1]
                return temp[1]*(temp[1]^2-(temp[2]*temp[3]))
            end
        else
            local a, b, c, d, e, f, g, h, i, xd, yd = unpack(iC)
            return function (x, y)
                local tx = x - xd
                local ty = y - yd
                return ty*(a + ty*(b + ty*c)) + tx*(d + ty*(e + ty*f) + tx*(g + ty*h +  tx*i))
            end
        end
    end
}

--## Calculate the gradient of the implicit function ##--
local calc_implicit_gradx = {
    [2] = function(p, iC)
        local temp = matmul({p}, iC)[1]
        return 2*temp[1]*iC[1][1]-(iC[1][2]*temp[3]+iC[1][3]*temp[2])
    end,
    [3] = function(p, iC)
        if #iC == 3 then
            local temp = matmul({p}, iC)[1]
            return 3*(temp[1]^2)*iC[1][1]-(iC[1][2]*temp[1]+iC[1][1]*temp[2])
        else
            local a, b, c, d, e, f, g, h, i, xd, yd = unpack(iC)
            local tx = p[1] - xd
            local ty = p[2] - yd
            return (d + ty*(e + ty*f) + 2*tx*(g + ty*h) +  3*(tx^2)*i)
        end
    end
}

--## Calculate gradient or second gradient of the parametric function ##--
local function get_grad(t, dC)
    local d = matmul(get_t_vec(t, #dC-1), dC)[1]
    if d[1]==0 and d[2]==0 and d[3]==0 then
        d = matmul(get_t_vec(t, #dC-1), derivate(dC))[1]
    end
    return d
end

--## get homogeneous plane that is the intersection of 2 tangent planes ##--
local function get_tg_intersection(dC, e1, e2)
    --## get the gradient in the point ##--
    local d1 = get_grad(e1[1], dC)
    local d2 = get_grad(e2[1], dC)

    local iC1 = get_plane(d1, e1[2])
    local iC2 = get_plane(d2, e2[2])
    return planes_intersection(iC1, iC2)
end

--## Make the monotone sample function ##--
local function monotone(dC, e1, e2, iC, hp, parsed)
    --## Compute the triangle box limits ##--
    local p1, p2 = e1[2], e2[2]
    local pp1, pp2 = project(p1), project(p2)
    local p3 = get_tg_intersection(dC, e1, e2)
    local y_lim = p3[2]

    local signal = 1
    if (pp2[2]-pp1[2])<0 then
        signal = -1
    end

    local L1, L2, L3 = left_line(p1, p2), {}, {}
    if signal == 1 then
        L2 = left_line(p1, p3)
        L3 = left_line(p3, p2)
    else
        L2 = left_line(p3, p2)
        L3 = left_line(p1, p3)
    end

    --## Compute additional parameters ##--
    local gradx = calc_implicit_gradx[#dC-1](hp, iC)
    local side = left_line(p1, p2, p3)
    local mon_type = get_monotone_type(pp1, pp2)
    if parsed.display["s"] then
        print("\n## Monotone ##")
        print("", "t ->", e1[1], e2[1])
        print("", "sig", signal, "side", side)
        print("", "p1", unpack(pp1))
        print("", "p2", unpack(pp2))
        print("", "p3", unpack(project(p3)))
        print("", "gradx", gradx)
        print("", "type", mon_type)
    end

    --## Prepare the implicit function ##--
    local implicit = calc_implicit[#dC-1](iC)

    local left = {}
    if gradx>0 then
        left = function(x, y)
            if implicit(x, y)<0 then return signal
            else return 0
            end
        end
    else
        left = function(x, y)
            if implicit(x, y)>0 then return signal
            else return 0
            end
        end
    end

    --## Fuse all ##--
    local triangle_bound = {}
    if side then
        triangle_bound = function(x, y)
            if L1(x, y)==0 then
                return 0
            else
                if y<y_lim then
                    tst = L2(x, y)~=0
                else
                    tst = L3(x, y)~=0
                end
                if tst then return signal
                else return left(x, y)
                end
            end
        end
    else
        triangle_bound = function(x, y)
            if L1(x, y)~=0 then
                return signal
            else
                if y<y_lim then
                    tst = L2(x, y)==0
                else
                    tst = L3(x, y)==0
                end
                if tst then return 0
                else return left(x, y)
                end
            end
        end
    end

    return triangle_bound, {mon_type, signal}
end

--## Find linear function root ##--
local function lin(b, a, parsed)
    if a == 0 then
        if parsed.display["s"] then
            print("", "a igual a 0")
        end
    else
        local t = -b/a
        if parsed.display["s"] then
            print("", "t:", t)
        end
        return t
    end
end

--## Find quadratic function roots ##--
local function root(c, b, a, parsed)
    if a==0 then
        if parsed.display["s"] then
            print("", "a==0")
        end
        return lin(c, b, parsed)
    end
    local delta = b^2-4*a*c
    if parsed.display["s"] then
        print("", "delta:", delta)
    end
    if math.abs(delta)<error_lim then
        local t = -b/(2*a)
        if parsed.display["s"] then
            print("", "t:", t)
        end
        return {t1}
    elseif delta > 0 then
        delta = math.sqrt(delta)
        local t1 = (-b+delta)/(2*a)
        local t2 = (-b-delta)/(2*a)
        if parsed.display["s"] then
            print("", "t1:", t1, "t2:", t2)
        end
        return {t1, t2}
    else
        if parsed.display["s"] then
            print("", "raiz complexa")
        end
    end
end

--## Calculate the implicit function ##--
local function quad_get_implicit(points, C, dC, calc, parsed)
    local p1 = points[1]
    local p2 = points[3]
    --## get the gradient in the point ##--
    local d1 = get_grad(0, dC)
    local d2 = get_grad(1, dC)

    --## get vectors perpendicular to planes (project the lines in the w=1) ##--
    local iC = {}
    iC[1] = get_plane(p1, p2)
    iC[2] = get_plane(d1, p1)
    iC[3] = get_plane(d2, p2)

    --## makes the iC adjustment ##--
    iC = transpose(iC)
    local temp = matmul({calc(0.5)}, iC)[1]
    local k2 = temp[1]^2
    local lm = temp[2]*temp[3]
    local a = k2/lm
    for i=1, 3 do
        iC[i][2] = iC[i][2]*a
    end

    if parsed.display["s"] then
        print("\n## Quadratic ##")
        print("", "d1", unpack(d1))
        print("", "d2", unpack(d2))
        print_mat(iC)
    end

    return iC
end

--## Calculate the implicit function ##--
local function cubic_get_implicit(points, C, dC, calc, parsed)
    --## Verify if it's a quadratic bezier ##--
    local tC = transpose(C)
    local kd, kc, kb, ka = unpack(tC[1])
    local ld, lc, lb, la = unpack(tC[2])

    local d = {0}
    for i=2, 4 do
        local temp = {}
        aa = {}
        for j=4, 1, -1 do
            if j~=i then
                temp[#temp+1] = C[j]
            end
        end
        d[i] = det3(temp)*(-1)^(i+1)
    end
    if parsed.display["s"] then
        print("\n## Cubic ##")
        print("d2:", d[2], "d3:", d[3], "d4:", d[4])
    end

    if math.abs(d[2])<error_lim and math.abs(d[3])<error_lim then
        if parsed.display["s"] then
            print("is a quadratic bezier")
        end
        return quad_get_implicit(points, C, dC, calc, parsed)
    end

    --## Display cubic type ##--
    if parsed.display["s"] then
        local delta = {}
        delta[1] = -d[2]^2
        delta[2] = d[2]*d[3]
        delta[3] = d[2]*d[4]-d[3]^2
        local discriminant = 4*delta[1]*delta[3]-delta[2]^2
        print("delta1:", delta[1], "delta2:", delta[2], "delta3:", delta[3])
        print("must be all 0: ", unpack(matmul({d}, C)[1]))
        print("", "discriminante", discriminant)

        local t = {}
        if discriminant < 0 then
            if parsed.display["s"] then
                print("", "Loop")
            end
            t = root(delta[3], delta[2], delta[1], parsed)
        else
            if parsed.display["s"] then
                if discriminant < 0 then
                    print("", "Cusp")
                else
                    print("", "Serpentine")
                end
                local a = -6*ka*lb + 6*kb*la
                local b = -6*ka*lc + 6*kc*la
                local c = -2*kb*lc + 2*kc*lb
                t = root(c, b, a, parsed)
                print("", "t ->", unpack(t))
            end
        end
    end

    --## Calculate the implicit function ##--
    local xd = points[1][1]
    local yd = points[1][2]
    local temp = {}
    for i=2, 4 do
        temp[#temp+1] = points[i][1]-xd
        temp[#temp+1] = points[i][2]-yd
    end
    local x1, y1, x2, y2, x3, y3 = unpack(temp)
    local a = -27*x1*x3^2*y1^2+81*x1*x2*x3*y1*y2-81*x1^2*x3*y2^2-81*x1*x2^2*y1*y3+54*x1^2*x3*y1*y3+81*x1^2*x2*y2*y3-27*x1^3*y3^2
    local b = 81*x1*x2^2*y1-54*x1^2*x3*y1-81*x1*x2*x3*y1+54*x1*x3^2*y1-9*x2*x3^2*y1-81*x1^2*x2*y2+162*x1^2*x3*y2-81*x1*x2*x3*y2+27*x2^2*x3*y2-18*x1*x3^2*y2+54*x1^3*y3-81*x1^2*x2*y3+81*x1*x2^2*y3-27*x2^3*y3-54*x1^2*x3*y3+27*x1*x2*x3*y3
    local c = -27*x1^3+81*x1^2*x2-81*x1*x2^2+27*x2^3-27*x1^2*x3+54*x1*x2*x3-27*x2^2*x3-9*x1*x3^2+9*x2*x3^2-x3^3
    local d = 27*x3^2*y1^3-81*x2*x3*y1^2*y2+81*x1*x3*y1*y2^2+81*x2^2*y1^2*y3-54*x1*x3*y1^2*y3-81*x1*x2*y1*y2*y3+27*x1^2*y1*y3^2
    local e = -81*x2^2*y1^2+108*x1*x3*y1^2+81*x2*x3*y1^2-54*x3^2*y1^2-243*x1*x3*y1*y2+81*x2*x3*y1*y2+27*x3^2*y1*y2+81*x1^2*y2^2+81*x1*x3*y2^2-54*x2*x3*y2^2-108*x1^2*y1*y3+243*x1*x2*y1*y3-81*x2^2*y1*y3-9*x2*x3*y1*y3-81*x1^2*y2*y3-81*x1*x2*y2*y3+54*x2^2*y2*y3+9*x1*x3*y2*y3+54*x1^2*y3^2-27*x1*x2*y3^2
    local f = 81*x1^2*y1-162*x1*x2*y1+81*x2^2*y1+54*x1*x3*y1-54*x2*x3*y1+9*x3^2*y1-81*x1^2*y2+162*x1*x2*y2-81*x2^2*y2-54*x1*x3*y2+54*x2*x3*y2-9*x3^2*y2+27*x1^2*y3-54*x1*x2*y3+27*x2^2*y3+18*x1*x3*y3-18*x2*x3*y3+3*x3^2*y3
    local g = -54*x3*y1^3+81*x2*y1^2*y2+81*x3*y1^2*y2-81*x1*y1*y2^2-81*x3*y1*y2^2+27*x3*y2^3+54*x1*y1^2*y3-162*x2*y1^2*y3+54*x3*y1^2*y3+81*x1*y1*y2*y3+81*x2*y1*y2*y3-27*x3*y1*y2*y3-27*x2*y2^2*y3-54*x1*y1*y3^2+18*x2*y1*y3^2+9*x1*y2*y3^2
    local h = -81*x1*y1^2+81*x2*y1^2-27*x3*y1^2+162*x1*y1*y2-162*x2*y1*y2+54*x3*y1*y2-81*x1*y2^2+81*x2*y2^2-27*x3*y2^2-54*x1*y1*y3+54*x2*y1*y3-18*x3*y1*y3+54*x1*y2*y3-54*x2*y2*y3+18*x3*y2*y3-9*x1*y3^2+9*x2*y3^2-3*x3*y3^2
    local i = 27*y1^3-81*y1^2*y2+81*y1*y2^2-27*y2^3+27*y1^2*y3-54*y1*y2*y3+27*y2^2*y3+9*y1*y3^2-9*y2*y3^2+y3^3
    local iC = {a, b, c, d, e, f, g, h, i, xd, yd}

    --## Display things ##--
    if parsed.display["s"] then
        print_vec(iC)
    end

    return iC
end

--## Redirect implicit commands ##--
local get_implicit = {
    [2] = quad_get_implicit,
    [3] = cubic_get_implicit
}

--## Process bezier curves ##--
local function left_bezier(points, n, raw_divs, parsed, segments, lims)
    --## generate parametric and the derivative matrix ##--
    local C = matmul(change_of_basis[n+1], points)
    calc = function(t)
        return matmul(get_t_vec(t, n), C)[1]
    end
    local dC = derivate(C)

    --## process the monotone divisions ##--
    local divs = {}
    for i=1, #raw_divs do
        divs[i] = raw_divs[i][1]
    end

    --## get implicit function ##--
    local iC = get_implicit[n](points, C, dC, calc, parsed)

    --## organize extremas ##--
    table.sort(divs)
    local extremas = {{0, points[1]}}
    for i = 1, #divs do
        local p = calc(divs[i])
        extremas[#extremas+1] = {divs[i], p}
    end

    extremas[#extremas+1] = {1, points[#points]}
    table.sort(extremas, function(a, b) return a[1]<b[1] end)

    --## generate monotones ##--
    local monotones = {}
    local hp = {}
    local mon_lim = {}
    local temp = 0
    for i=2, #extremas do
        if extremas[i-1][1]~=extremas[i][1] then
            hp = calc((extremas[i-1][1]+extremas[i][1])/2)
            temp = {get_box_lim({extremas[i-1][2], extremas[i][2]})}
            monotones[#monotones+1], temp[#temp+1] = monotone(dC, extremas[i-1],
                                                              extremas[i], iC, hp, parsed)
            mon_lim[#mon_lim+1] = temp
        end
    end

    --## bounding function ##--
    for i=1, #monotones do
        segments[#segments+1] = monotones[i]
        lims[#lims+1] = mon_lim[i]
    end
end

--## Redirect commands ##--
local get_path_segments = {
    line = function(segments, command, parsed, lims)
        if command[1][2]~=command[2][2] then
            segments[#segments+1], lims[#lims+1] = left_line(command[1], command[2])
        end
    end,

    quadratic = function(segments, command, parsed, lims)
        left_bezier({command[1], command[2], command[3]},
                     2, command["div"], parsed, segments, lims)
    end,

    cubic = function(segments, command, parsed, lims)
        left_bezier({command[1], command[2],
                     command[3], command[4]},
                     3, command["div"], parsed, segments, lims)
    end,
}

--## Make the monotone rasterisation ##--
local function get_segment_buffer(segment, y_lim, x_lim, index, info, nr)
    -- initialize variables and precompute
    local monotone_type, signal = unpack(info)
    local i, j = 0, 0
    local j_lim = floor((y_lim[2]-y_lim[1])*nr+0.5)
    local x, y = x_lim[1], y_lim[1]
    local condition = function(r) return r<=j_lim end
    if monotone_type==-1 then
        condition = function(r) return r>=0 end
        j = j_lim
        y = y_lim[1]+j/nr
    end
    local buffer = {}

    -- rasterize
    while(condition(j)) do
        if segment(x, y)==0 then
            buffer[index[2]+j] = {index[1]+i, signal}
            j = j+monotone_type
            y = y_lim[1]+j/nr
        else
            i = i+1
            x = x_lim[1]+i/nr
        end
    end
    return buffer
end

--## Main path ##--
local function get_path_acc(shape, scene_proj, parsed, nr, calc_index)
    -- path processing
    local transf = scene_proj*shape:get_xf()
    local begin = {}
    local segments = {}
    local lims = {}
    local pdata = 0
    if shape:get_type() == shape_type.path then
        pdata = shape:get_path_data()
    else
        pdata = shape:as_path_data()
    end
    local div_buffer = {}
    pdata:iterate(filter.make_input_path_f_xform(transf,
        filter.make_input_path_f_downgrade_degenerates(error_lim,
            filter.make_input_path_f_find_monotonic_parameters(
                filter.make_input_path_f_find_cubic_parameters({
            root_dx_parameter = function(self, t)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "root_dx_parameter", t) end
                div_buffer[#div_buffer+1] = {t, "x"}
            end,

            root_dy_parameter = function(self, t)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "root_dy_parameter", t) end
                div_buffer[#div_buffer+1] = {t, "y"}
            end,

            inflection_parameter = function(self, t)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "inflection_parameter", t) end
                div_buffer[#div_buffer+1] = {t, "i"}
            end,

            double_point_parameter = function(self, t)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "double_point_parameter", t) end
                div_buffer[#div_buffer+1] = {t, "d"}
            end,

            begin_contour = function(self, x0, y0)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "begin_contour", x0, y0) end
                begin = {x0, y0, 1}
            end,

            end_open_contour = function(self, x0, y0)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "end_open_contour", x0, y0) end
                local command = {{x0, y0, 1}, begin}
                get_path_segments["line"](segments, command, parsed, lims)
            end,

            end_closed_contour = function(self, x0, y0)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "end_closed_contour", x0, y0) end
            end,

            linear_segment = function(self, x0, y0, x1, y1)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "linear_segment", x0, y0, x1, y1) end
                local command = {{x0, y0, 1}, {x1, y1, 1}}
                get_path_segments["line"](segments, command, parsed, lims)
            end,

            quadratic_segment = function(self, x0, y0, x1, y1, x2, y2)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "quadratic_segment", x0, y0, x1, y1, x2, y2) end
                local command = {{x0, y0, 1}, {x1, y1, 1}, {x2, y2, 1},
                                 div=div_buffer}
                get_path_segments["quadratic"](segments, command, parsed, lims)
                div_buffer = {}
            end,

            cubic_segment = function(self, x0, y0, x1, y1, x2, y2, x3, y3)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n","cubic_segment", x0, y0, x1, y1, x2, y2, x3, y3) end
                local command = {{x0, y0, 1}, {x1, y1, 1}, {x2, y2, 1}, {x3, y3, 1},
                                 div=div_buffer}
                get_path_segments["cubic"](segments, command, parsed, lims)
                div_buffer = {}
            end,

            rational_quadratic_segment = function(self, x0, y0, x1, y1, w1,
                x2, y2)
                if parsed.display[s] and not parsed.display[r] then
                    print("\n", "rational_quadratic_segment", x0, y0, x1, y1, w1,
                          x2, y2) end
                local command = {{x0, y0, 1}, {x1, y1, w1}, {x2, y2, 1},
                                 div=div_buffer}
                get_path_segments["quadratic"](segments, command, parsed, lims)
                div_buffer = {}
            end,
    })))))

    --if don't have any segment
    if #lims==0 then
        return nil
    end

    -- get the minimum and maximum coordinate
    local calc_min_max = function(c_min, c_max)
        return {(ceil(c_min*nr-0.5)+0.5)/nr, (ceil(c_max*nr-0.5)-0.5)/nr}
    end

    -- prepare the buffer generating function
    local function gen_buffer()
        local buffer = winding_buffer()
        for n=1, #segments do
            local x_lim = calc_min_max(lims[n][1], lims[n][2])
            local y_lim = calc_min_max(lims[n][3], lims[n][4])
            local index = calc_index(x_lim[1], y_lim[1])
            buffer:add(get_segment_buffer(segments[n], y_lim, x_lim, index, lims[n][5], nr))
        end
        return buffer:get()
    end
    return gen_buffer
end

------------------------------------
--             Colors             --
------------------------------------

--## Premultiply opacity in alpha ##--
local function apply_opacity(color, opacity)
    local rgba = {unpack(color)}
    rgba[4] = rgba[4]*opacity
    return rgba
end

--## Premultiply alpha ##--
local function multiply_alpha(color)
    local rgba = {unpack(color)}
    if rgba[4]==nil then
        rgba[4] = 1
        return rgba
    end
    for j=1, 3 do
        rgba[j] = rgba[j]*rgba[4]
    end
    return rgba
end

--## Make the interpolation of colors c1 and c2 ##--
local function intepolate(p1, p2, d, c1, c2, k)
    -- p1 and p2 is the parameter values of the colors c1 and c2
    -- d is the precalculated distance between the two colors
    -- k is the interpolation parameter value
    local C = {}
    local r1 = (p2-k)/d
    local r2 = (k-p1)/d
    for j=1, #c1 do
        C[j] = c1[j]*r1+c2[j]*r2
    end
    return C
end

--## Adjust the parameter depending on the spread method ##--
local get_ramp_acc = {
    [spread.clamp] = function(index, colors, img)
        --print("spread.clamp")
        local ramp = {}
        if not img then
            ramp = function(k)
                local t = b_search(index, k)
                return colors[t](k)
            end
        else
            local w = img:get_width()
            local h = img:get_height()
            ramp = function(x, y)
                local temp = {}
                if x < 1 then     x = 1
                elseif x > w then x = w end
                if y < 1 then     y = 1
                elseif y > h then y = h end
                return {img:get_pixel(x, y)}
            end
        end
        return ramp
    end,

    [spread.wrap] = function(index, colors, img)
        --print("spread.wrap")
        --color[0] = function(k) error(k) end
        --color[#color] = function(k) error(k) end
        local ramp = {}
        if not img then
            ramp = function(k)
                local k_mod = floor(k)
                k = k-k_mod
                local t = b_search(index, k)
                return colors[t](k)
            end
        else
            local w = img:get_width()
            local h = img:get_height()
            ramp = function(x, y)
                x = (x-1)%w+1
                y = (y-1)%h+1
                return {img:get_pixel(x, y)}
            end
        end
        return ramp
    end,

    [spread.mirror] = function(index, colors, img)
        --print("spread.mirror")
        --color[0] = function(k) error(k) end
        --color[#color] = function(k) error(k) end
        local ramp = {}
        if not img then
            ramp = function(k)
                local k_mod = floor(k)
                local mod_2 = k_mod%2
                k = (k-k_mod)*(1-2*mod_2)+mod_2
                local t = b_search(index, k)
                return colors[t](k)
            end
        else
            local w = img:get_width()-1
            local h = img:get_height()-1
            ramp = function(x, y)
                x = x-1
                y = y-1
                local rx = x%w
                local tx = ((x-rx)/w)%2
                local ry = y%h
                local ty = ((y-ry)/h)%2
                x = rx*(1-2*tx)+tx*w+1
                y = ry*(1-2*ty)+ty*h+1
                return {img:get_pixel(x, y)}
            end
        end
        return ramp
    end,

    [spread.transparent] = function(index, colors, img)
        --print("spread.transparent")
        local ramp = {}
        if not img then
            ramp = function(k)
                if k<0 or k>1 then
                    return {0, 0, 0, 0}
                end
                local t = b_search(index, k)
                return colors[t](k)
            end
        else
            local w = img:get_width()
            local h = img:get_height()
            local nothing = {0, 0, 0, 0}
            local complete = function(a, b, c, d) return {a, b, c, d} end
            if img:get_num_channels() ~= 4 then
                complete = function(a, b, c) return {a, b, c, 1} end
            end
            ramp = function(x, y)
                local temp = {}
                if x < 1 or x > w or y < 1 or y > h then
                    return nothing
                else
                    return complete(img:get_pixel(x, y))
                end
            end
        end
        return ramp
    end,
}

--## Organize the stops, make the color generating function and organize the intervals ##--
local function get_intervals(ramp, opacity)
    local stops = {}
    local index = {}
    local colors = {}
    local repeated = 0
    local i = 1
    local a = 2

    -- make the color generating functions and save the intervals
    for _, stop in ipairs(ramp:get_color_stops()) do
        stops[i] = {stop:get_offset(), stop:get_color()}

        -- initialize (the first color)
        if i==1 then
            index[1] = stops[1][1]
            local pre_alpha1 = multiply_alpha(apply_opacity(stops[1][2], opacity))
            colors[1] = function(k)
                            return pre_alpha1 end

        -- eliminate middle points in conflict
        elseif stops[i][1]==stops[i-1][1] then
            if repeated == 1 then
                stops[i-1] = stops[i]
                i = i-1
            else
                repeated = repeated+1
            end

        -- make the middle colors generating functions
        else
            repeated = 0
            if index[a-1]~=stops[i][1] then
                index[a] = stops[i][1]
                local p1 = index[a-1]
                local p2 = index[a]
                local d = (p2-p1)
                local c1 = apply_opacity(stops[i-1][2], opacity)
                local c2 = apply_opacity(stops[i][2], opacity)
                colors[a] = function(k)
                    return multiply_alpha(intepolate(p1, p2, d, c1, c2, k))
                end
                a = a+1
            end
        end
        i = i+1
    end

    -- (the last color)
    local pre_alpha2 = multiply_alpha(apply_opacity(stops[#stops][2], opacity))
    colors[#colors+1] = function(k)
                             return pre_alpha2 end

    return index, colors
end

--## Main color ##
local get_paint_acc = {
    [paint_type.solid_color] = function(paint, T, parsed)
        local rgba = multiply_alpha(apply_opacity(paint:get_solid_color(),
                                                  paint:get_opacity()))
        local function get_color(x, y)
            return {unpack(rgba)}
        end
        return get_color
    end,

    [paint_type.linear_gradient] = function(paint, T, parsed)
        local Ti = (T*paint:get_xf()):inverse()
        local lg = paint:get_linear_gradient_data()

        --## Color processing ##--
        local opacity = paint:get_opacity()
        local raw_color_ramp = lg:get_color_ramp()
        local index, colors = get_intervals(raw_color_ramp, opacity)
        local color_ramp = get_ramp_acc[raw_color_ramp:get_spread()](index, colors)

        --## Position processing ##--
        -- move p1 to origin --
        -- scale p2 to make the distance from origin equals 1 --
        -- rotate p2 to y = 1 --
        local p1_x, p1_y = lg:get_x1(), lg:get_y1()
        local p2_x, p2_y = lg:get_x2(), lg:get_y2()
        local v_x, v_y = p2_x - p1_x, p2_y - p1_y
        local d2 = v_x^2 + v_y^2
        local a, b, c = unpack(_M.affinity(v_x/d2,  v_y/d2, (-v_x*p1_x-v_y*p1_y)/d2, 0, 1, 0)*Ti)

        local function get_color(x, y)
            return color_ramp(a*x + b*y + c)
        end
        return get_color
    end,

    [paint_type.radial_gradient] = function(paint, T, parsed)
        local Ti = (T*paint:get_xf()):inverse()
        local g = paint:get_radial_gradient_data()

        --## Color processing ##--
        local opacity = paint:get_opacity()
        local raw_color_ramp = g:get_color_ramp()
        local index, colors = get_intervals(raw_color_ramp, opacity)
        local color_ramp = get_ramp_acc[raw_color_ramp:get_spread()](index, colors)

        --## Position processing ##--
        -- move focal point to origin --
        local f_x, f_y = g:get_fx(), g:get_fy()
        local transf = _M.translation(-f_x, -f_y)

        local c_x, c_y = transf:apply(g:get_cx(), g:get_cy())
        local r = g:get_r()
        local get_color = {}
        -- Rescale and rotate center to (-1, 0) if isn't in origin --
        if c_x == 0 and c_y == 0 then --if center is in origin
            transf = transf:scaled(1/r)*Ti --make radius equals 1
            get_color = function (x, y) --distance of the point from origin
                x, y = transf:apply(x, y)
                k = math.sqrt(x*x+y*y)
                return color_ramp(k)
            end
            return get_color
        else --Rescale and rotate center to (-1, 0)
            local d = c_x*c_x+c_y*c_y
            transf = _M.linearity(-c_x/d, -c_y/d, c_y/d, -c_x/d)*transf
            r = r/math.sqrt(d)

            -- precalculate constant values --
            local rp = (1-r*r)
            transf = transf*Ti

            if abs(rp)<error_lim^2 then -- if the focus is in the circunference
                -- generate the mean color
                local mean_color = {0, 0, 0, 0}
                for i=1, #index-1 do
                    print(index[i+1]+index[i])
                    local temp_color = color_ramp((index[i]+index[i+1])/2)
                    for c=1, #temp_color do
                        mean_color[c] = mean_color[c]+temp_color[c]*(index[i+1]-index[i])
                    end
                end
                get_color = function (x, y)
                    x, y = transf:apply(x, y)
                    if x>=0 then return mean_color
                    else
                        local P = 1+(y*y)/(x*x)
                        local d = math.sqrt(1-P*rp)
                        k = -x*P/2
                    end
                    return color_ramp(k)
                end
            else  -- if the focus is inside the circle
                local rp1 = 1/rp
                local srp = math.sqrt(-rp)

                get_color = function (x, y)
                    x, y = transf:apply(x, y)
                    if abs(x)<error_lim then
                        k = abs(y)/srp
                    else
                        local P = 1+(y*y)/(x*x)
                        local d = math.sqrt(1-P*rp)
                        if x>0 then
                            k = -x*(1+d)*rp1
                        else
                            k = -x*(1-d)*rp1
                        end
                    end
                    return color_ramp(k)
                end
            end
        end
        return get_color
    end,

    [paint_type.texture] = function(paint, T, parsed)
        local tex = paint:get_texture_data()
        local img = tex:get_image()
        local transf = _M.translation(0.5, 0.5)*_M.scaling(img:get_width(), img:get_height())*(T*paint:get_xf()):inverse()
        local opacity = paint:get_opacity()
        local color_ramp = get_ramp_acc[tex:get_spread()](nil, nil, img)

        local a_channel = img:get_num_channels()==4
        local function get_color(x, y)
            x, y = transf:apply(x, y)
            local xp = {floor(x), ceil(x)}
            local yp = {floor(y), ceil(y)}
            local xk = x-xp[1]
            local yk = y-yp[1]
            local c1 = color_ramp(xp[1], yp[1])
            local c2 = color_ramp(xp[2], yp[1])
            local c3 = color_ramp(xp[1], yp[2])
            local c4 = color_ramp(xp[2], yp[2])

            local c12 = intepolate(0, 1, 1, c1, c2, xk)
            local c34 = intepolate(0, 1, 1, c3, c4, xk)
            local fcolor = intepolate(0, 1, 1, c12, c34, yk)

            if not a_channel then
                fcolor[4] = 1
            end
            return multiply_alpha(apply_opacity(fcolor, opacity))
        end
        return get_color
    end,
}

------------------------------------
--         Main functions         --
------------------------------------

local function parse(args)
    -- Aqui vc tem que colocar as variáveis que vc vai usar.
    --quando vc quiser usar uma das variáveis, é essa tabela que vc vai estar usando.
	local parsed = {
		pattern = nil,
		tx = nil,
		ty = nil,
        linewidth = nil,
		maxdepth = nil,
		p = nil,
		dumptreename = nil,
		dumpcellsprefix = nil,
		display = {}
	}
	-- Esta tabela é onde será processada a string digitada
	-- Ex.: luapp process.lua render imagem.rvg imagem.png -key1 -key2:10 -key3:abc
    local options = {
        { "^(%-tx:(%-?%d+)(.*))$", function(all, n, e)
            if not n then return false end
            assert(e == "", "trail invalid option " .. all)
            parsed.tx = assert(tonumber(n), "number invalid option " .. all)
            return true
        end },
        { "^(%-ty:(%-?%d+)(.*))$", function(all, n, e)
            if not n then return false end
            assert(e == "", "trail invalid option " .. all)
            parsed.ty = assert(tonumber(n), "number invalid option " .. all)
            return true
        end },
        { "^(%-p:(%d+)(.*))$", function(all, n, e)
            if not n then return false end
            assert(e == "", "trail invalid option " .. all)
            parsed.p = assert(tonumber(n), "number invalid option " .. all)
            return true
        end },
        { "^(%-display:(.*))$", function(all, c)
            if not c then return false end
            print("display arguments")
                print("p", ":display paths")
                print("s", ":display shapes")
                print("r", ":display raw data (dump)")
            local temp = {}
            c:gsub(".",function(char) table.insert(temp, char) end)
            parsed.display = {}
            for i=1, #temp do
                parsed.display[temp[i]] = true
            end
            return true
        end },
        { ".*", function(all)
            error("unrecognized option " .. all)
        end }
    }

    -- process options
    for i, arg in ipairs(args) do
        for j, option in ipairs(options) do
            if option[2](arg:match(option[1])) then
                break
            end
        end
    end
    return parsed
end

local winding_rules = {
    [_M.winding_rule.odd] = function (a)
        return a%2==1
    end,
    [_M.winding_rule.non_zero] = function (a)
        return a~=0
    end,
}

local nr = 1
local function super_sample(img, accelerated, width, height,
                            vxmin, vymin, vxmax, vymax)
    --## Precompute constants ##--
    local nr2 = nr^2
    local max_i, max_j = unpack(accelerated.calc_index(vxmax, vymax))
    max_i, max_j = max_i-1, max_j-1
    local pre_calc_x = {}
    for i=max_i, 1, -1 do
        pre_calc_x[i] = (i-0.5)/nr+vxmin
    end

    --## Generate all buffers and organize by y value ##--
    local buffers = {}
    local w_rules = {}
    local time = chronos.chronos()
    stderr("\tGenerating Buffers\t")
    for n = #accelerated, 1, -1 do
        stderr("\r%5g%%", floor(1000*n/#accelerated)/10)
        local buff = accelerated[n].buffer()
        for j, values in pairs(buff) do
            if buffers[j] == nil then
                buffers[j] = {}
            end
            buffers[j][n] = values
        end
        w_rules[n] = winding_rules[accelerated[n].winding_rule]
    end
    stderr("\tBuffer generated in %.3fs\n", time:elapsed())
    time:reset()

    --## Sampling ##--
    local temp_img = {}
    stderr("\tSampling\t")
    for j=1, max_j do
        stderr("\r%5g%%", floor(1000*j/max_j)/10)

        -- ## Precompute constants an initialize variables ##--
        local lin_buff = buffers[j]
        local sums = {}
        local shapes = {}
        local next_i = 0
        if lin_buff then
            -- list the active shapes in the line j
            for n, temp_buffer in pairs(lin_buff) do
                sums[n] = 0
                shapes[#shapes+1] = n
                next_i = max(next_i, temp_buffer[#temp_buffer][1])
            end
            table.sort(shapes)
        end

        local y = (j-0.5)/nr+vymin
        local pixel_y = ceil(y)-vymin
        if not temp_img[pixel_y] then temp_img[pixel_y] = {} end

        local painting_shapes = {}
        for i=max_i, 1, -1 do
            --## Verify the changes in the line states ##--
            if next_i>i then
                next_i = 0
                painting_shapes = {}
                local new_shapes = {}
                --iterate over the active shapes
                for _, n in ipairs(shapes) do
                    while #lin_buff[n]~=0 do
                        if lin_buff[n][#lin_buff[n]][1]>i then
                            sums[n] = sums[n]+table.remove(lin_buff[n])[2]
                        else
                            next_i = max(next_i, lin_buff[n][#lin_buff[n]][1])
                            break
                        end
                    end
                    -- if the shape still active
                    if #lin_buff[n]~=0 or sums[n]~=0 then
                        new_shapes[#new_shapes+1] = n
                        -- if the shape must be painted in the pixel
                        if w_rules[n](sums[n]) then
                            painting_shapes[#painting_shapes+1] = n
                        end
                    end
                end
                -- update the active shapes
                shapes = new_shapes
            end

            -- ## Samples necessary shapes ##--
            local x = pre_calc_x[i]
            local color = PixelColor()
            local stop = false
            -- iterate over the shapes that must be painted
            for ni=#painting_shapes, 1, -1 do
                stop = color:blend(accelerated[painting_shapes[ni]].paint(x, y))
                if stop then
                    break
                end
            end
            -- if isn't already opaque
            if not stop then
              color:blend(background:get_solid_color())
            end
            local c = {color:unpack()}
            local pixel_x = ceil(x)-vxmin
            if not temp_img[pixel_y][pixel_x] then
                temp_img[pixel_y][pixel_x] = c
            else
                for k=1, 4 do
                    temp_img[pixel_y][pixel_x][k] = temp_img[pixel_y][pixel_x][k]+c[k]
                end
            end
        end
    end
    stderr("\tSampled in %.3fs\n", time:elapsed())
    time:reset()
    stderr("\tPost processing\t")
    for y = 1, height do
        stderr("\r%5g%%", floor(1000*y/height)/10)
        for x = 1, width do
            img:set_pixel(x, y, apply_gamma(temp_img[y][x], nr2))
        end
    end
    stderr("\tPost processed in %.3fs\n", time:elapsed())
    time:reset()
end

function _M.accelerate(scene, window, viewport, args)
    local parsed = parse(args)
    scene = scene:windowviewport(window, viewport)
    local scene_proj = scene:get_xf()

    -- calculate the buffer index
    local vxmin, vymin = unpack(viewport, 1, 4)
    local accel = {}
    accel.calc_index = function(x, y)
        return {floor((x-vxmin)*nr+1), floor((y-vymin)*nr+1)} end

    local transforms = TransformStack()
    local i = 1
    scene:get_scene_data():iterate{
        painted_shape = function(self, winding_rule, shape, paint)
            local T = scene_proj*transforms:top()
            if parsed.display["r"] then
                print("--## Painted shape", i, "##--")
                print(winding_rule)
                print(paint, paint:get_xf())
                print_paint[paint:get_type()](paint)
            end

            local tpaint = get_paint_acc[paint:get_type()](paint, T, parsed)
            if parsed.display["r"] then
                print(shape, shape:get_xf())
                print_shape[shape:get_type()](shape)
            end
            local buffer = get_path_acc(shape, T, parsed, nr, accel.calc_index)

            if buffer then
                accel[i] = {paint = tpaint, buffer = buffer, winding_rule = winding_rule}
                i = i + 1
            end
        end,
        begin_transform = function(self, depth, xf)
            transforms:push(xf)
        end,
        end_transform = function(self, depth, xf)
            transforms:pop()
        end,
    }
    accel["parsed"] = parsed
    return accel
end

function _M.render(accelerated, window, viewport, file, args)
    local parsed = parse(args)
    stderr("parsed arguments\n")
    for i,v in pairs(parsed) do
        if type(v) == "table" then
            local temp = {}
            for k, _ in pairs(v) do temp[#temp+1] = k end
            print(table.concat({"  -", tostring(i)}), unpack(temp))
        else
            stderr("  -%s:%s\n", tostring(i), tostring(v))
        end
    end
    local time = chronos.chronos()
    -- Get viewport to compute pixel centers
    local vxmin, vymin, vxmax, vymax = unpack(viewport, 1, 4)
    local width, height = vxmax-vxmin, vymax-vymin
    assert(width > 0, "empty viewport")
    assert(height > 0, "empty viewport")
    -- Allocate output image
    local img = image.image(width, height, 4)
    -- Render
    super_sample(img, accelerated, width, height, vxmin, vymin, vxmax, vymax)
    stderr("\n")
    stderr("rendering in %.3fs\n", time:elapsed())
    time:reset()
        -- Store output image
        image.png.store8(file, img)
    stderr("saved in %.3fs\n", time:elapsed())
end

return _M
