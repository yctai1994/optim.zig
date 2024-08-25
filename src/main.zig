pub fn main() void {}

pub const Info = enum {
    full_kkt,
    trunc_kkt,
    kkt_terminated,
    full_quasi,
    trunc_quasi,
    quasi_terminated,
    dogleg,
    xtol_terminated,
    gtol_terminated,
};

test "test" {
    const page = std.testing.allocator;
    const ArrF64 = Array(f64){ .allocator = page };

    const log_file: std.fs.File = try std.fs.cwd().createFile("../logs/log.txt", .{ .read = false });
    defer log_file.close();

    const writer = log_file.writer();
    try writer.writeAll("# xk[0], xk[1], pk[0], pk[1], xn[0], xn[1], step-size, info\n");

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    // Setup
    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    var lb: [2]f64 = .{ -2.0, -1.0 };
    var ub: [2]f64 = .{ 0.8, 1.0 };

    var xk: [2]f64 = undefined;
    var gk: [2]f64 = undefined;
    var pk: [2]f64 = undefined;

    var tk: [2]f64 = undefined; // breakpoints' step sizes
    var ts: [2]f64 = undefined;

    var xt: [2]f64 = undefined;
    var xc: [2]f64 = undefined;
    var Wk: [2]usize = undefined;
    var ta: usize = undefined;

    var pN: [2]f64 = undefined;
    var pS: [2]f64 = undefined;
    var sz: f64 = undefined;

    var xn: [2]f64 = undefined;
    var gn: [2]f64 = undefined;

    var sk: [2]f64 = undefined;
    var yk: [2]f64 = undefined;
    var ak: [2]f64 = undefined;

    var uk: [2]f64 = undefined;
    var vk: [2]f64 = undefined;
    var bf: [2]f64 = undefined; // buffer

    const Bk: [][]f64 = try ArrF64.matrix(2, 2); // Bk = Rᵀ⋅R = cholescky( exact Hessian )
    defer ArrF64.free(Bk);

    const Ck: [][]f64 = try ArrF64.matrix(2, 2); // Ck = Ak⋅Hk⋅Akᵀ
    defer ArrF64.free(Ck);

    var sn: f64 = undefined;
    var ym: f64 = undefined;
    var rk: f64 = undefined;

    var info: Info = undefined;

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    // Initialization
    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    const rosenbrock: Rosenbrock = .{ .a = 1.0, .b = 100.0 };

    inline for (&xk, .{ -1.2, 1.0 }) |*p, v| p.* = v;

    rosenbrock.grad(&xk, &gk);

    for (Bk, 0..) |Bk_i, i| {
        @memset(Bk_i, 0.0);
        Bk_i[i] = 1.0;
    }

    var kx: usize = 0;

    try writer.print("# lb = {e: >.5}\n# ub = {e: >.5}\n# x0 = {e: >.5}\n", .{ lb, ub, xk });

    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    // Iteration
    // = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

    iteration: while (kx < 300) : (kx += 1) {
        sz = std.math.inf(f64);

        try project(&lb, &ub, &xk, &gk, Bk, &pk, &tk, &ts, &xt, &xc, &Wk, &ta);

        if (ta != 0) {
            try solveKKT(&xk, &gk, Bk, &pk, &xt, &xc, Ck, &Wk, ta);

            search(rosenbrock, &xk, &gk, &pk, &xn, &gn, 1e-2, 1.0, .{}) catch |err| {
                switch (err) {
                    WolfeErrors.DescentDirectionError => {
                        info = .kkt_terminated;
                        break :iteration;
                    },
                    else => return err,
                }
            };

            truncate(&lb, &ub, &xn, &Wk, &ta);

            if (ta != 0) {
                rosenbrock.grad(&xn, &gn); // gn ← ∇f(xn)
                info = .trunc_kkt;
            } else {
                info = .full_kkt;
            }
        } else {
            for (&lb, &ub, &xk) |lb_i, ub_i, xk_i| {
                sz = @min(sz, @abs(lb_i - xk_i));
                sz = @min(sz, @abs(ub_i - xk_i));
            }

            if (0 < sz) {
                try dogleg(Bk, &gk, &pk, &pN, &pS, sz);
                for (&xn, &xk, &pk) |*xn_i, xk_i, pk_i| xn_i.* = xk_i + pk_i;
                rosenbrock.grad(&xn, &gn);
                info = .dogleg;
            } else {
                // pk ← -Bk⁻¹⋅∇f(xk)
                for (&pk, &gk) |*pk_i, gk_i| pk_i.* = -gk_i;
                try trsv('R', 'T', Bk, &pk);
                try trsv('R', 'N', Bk, &pk);

                // xn ← xk + α⋅pk, gn ← ∇f(xn)
                search(rosenbrock, &xk, &gk, &pk, &xn, &gn, null, null, .{}) catch |err| {
                    switch (err) {
                        WolfeErrors.DescentDirectionError => {
                            info = .quasi_terminated;
                            break :iteration;
                        },
                        else => return err,
                    }
                };

                truncate(&lb, &ub, &xn, &Wk, &ta);

                if (ta != 0) {
                    rosenbrock.grad(&xn, &gn); // gn ← ∇f(xn)
                    info = .trunc_quasi;
                } else {
                    info = .full_quasi;
                }
            }
        }

        // sk ← xn - xk
        for (&sk, &xn, &xk) |*sk_i, xn_i, xk_i| sk_i.* = xn_i - xk_i;
        sn = 0.0;
        for (&sk) |sk_i| sn += pow2(sk_i);
        sn = @sqrt(sn);

        if (sn <= 1e-16) {
            info = .xtol_terminated;
            break :iteration;
        }

        // yk ← ∇f(xn) - ∇f(xk) = gn - gk
        for (&yk, &gn, &gk) |*yk_i, gn_i, gk_i| yk_i.* = gn_i - gk_i;
        ym = 0.0;
        for (&yk) |ym_i| ym = @max(ym, @abs(ym_i));

        if (ym <= 1e-12) {
            info = .gtol_terminated;
            break :iteration;
        }

        // secant_norm2 ← skᵀ⋅yk
        const secant_norm2: f64 = try dot(&sk, &yk);
        rk = secant_norm2 / try dot(&yk, &yk); // γ = sᵀ⋅y / yᵀ⋅y

        if (2.2e-16 < rk) { // strong curvature condition
            // sk ← Lkᵀ⋅sk = Rk⋅sk
            try trmv('R', 'N', Bk, &sk);

            // skᵀ⋅(Lk⋅Lkᵀ)⋅sk ← skᵀ⋅sk
            const quadratic_form: f64 = try dot(&sk, &sk);

            // αk ← √(secant_norm2 / quadratic_form)
            const alpha_k: f64 = @sqrt(secant_norm2 / quadratic_form);

            // ak ← αk⋅Lkᵀ⋅sk = αk⋅Rk⋅sk
            for (&ak, &sk) |*ak_i, sk_i| ak_i.* = alpha_k * sk_i;

            // ‖ak‖ ← √(skᵀ⋅yk)
            const secant_norm: f64 = @sqrt(secant_norm2);

            // uk ← ak / ‖ak‖
            for (&uk, &ak) |*uk_i, ak_i| uk_i.* = ak_i / secant_norm;

            // vk ← Lk⋅ak = Rkᵀ⋅ak
            for (&vk, 0..) |*vk_i, i| {
                vk_i.* = 0.0;
                for (0..i + 1) |j| vk_i.* += Bk[j][i] * ak[j];
            }

            // vk ← (yk - Rkᵀ⋅ak) / ‖ak‖
            for (&vk, &yk) |*vk_i, yk_i| vk_i.* = (yk_i - vk_i.*) / secant_norm;

            try update(Bk, &uk, &vk, &bf);
        }

        try writer.print(
            "{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{any}\n",
            .{ xk[0], xk[1], pk[0], pk[1], xn[0], xn[1], sz, info },
        );

        @memcpy(&xk, &xn);
        @memcpy(&gk, &gn);
    }

    try writer.print(
        "{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{e: >.5}\t{any}\n",
        .{ xk[0], xk[1], pk[0], pk[1], xn[0], xn[1], sz, info },
    );
}

fn dot(x: []f64, y: []f64) !f64 {
    if (x.len != y.len) return error.DimensionMismatch;
    var t: f64 = 0.0;
    for (x, y) |x_i, y_i| t += x_i * y_i;
    return t;
}

inline fn pow2(x: f64) f64 {
    return x * x;
}

const std = @import("std");
const testing = std.testing;

const trmv = @import("./linalg/trmv.zig").trmv;
const trsv = @import("./linalg/trsv.zig").trsv;

const update = @import("./linalg/cholesky.zig").update;

const checkbounds = @import("./active.zig").checkbounds;
const truncate = @import("./active.zig").truncate;
const project = @import("./active.zig").project;
const solveKKT = @import("./active.zig").solveKKT;
const dogleg = @import("./dogleg.zig").dogleg;

const search = @import("./wolfe.zig").search;
const WolfeErrors = @import("./wolfe.zig").Errors;

const Array = @import("./linalg/array.zig").Array;
const Rosenbrock = @import("./misc/Rosenbrock.zig");
