# ═══════════════════════════════════════════════════════════════════════════════
# Plot the cross-language benchmark results
# ═══════════════════════════════════════════════════════════════════════════════
#
# Reads the CSV written by `benchmark_crosslang.jl` and draws every case in one
# figure: one log-log panel per (K, D) case, one line per backend.
#
# Usage:
#   julia --project=benchmark benchmark/plot_crosslang.jl                        # results/benchmark_timings_latest.csv
#   julia --project=benchmark benchmark/plot_crosslang.jl path/to/timings.csv
#
# Output (into benchmark/, which is where `docs/src/benchmarks.md` links from):
#   timing_crosslang.svg              absolute times, all cases
#   timing_crosslang_ratio.svg        the same panels as a ratio to ExpectationMaximization.jl
#   timing_crosslang_<date>.svg       dated copies of both, so a rerun does not
#   timing_crosslang_ratio_<date>.svg overwrite the figure the docs point at
#
# Plotting is deliberately a separate script from the benchmark: the previous one
# (`benchmark_v2_K2_unidim.jl`) kept its timings in a `Dict` in memory, so
# restyling a figure meant rerunning all four backends.

cd(@__DIR__)
import Pkg;
Pkg.activate(".");

ENV["GKSwstype"] = "100"   # headless GR, for CI

using Dates
using LaTeXStrings
using Printf
using Statistics: median
using StatsPlots

const CSV_DEFAULT = joinpath("results", "benchmark_timings_latest.csv")
const REFERENCE = "ExpectationMaximization.jl"

# ── Reading the results ───────────────────────────────────────────────────────
#
# The column order is fixed by `benchmark_crosslang.jl`: case,backend,K,D,N,time_s.
# No quoting and no embedded commas, so `split` is enough and the benchmark
# environment needs no CSV.jl.

struct Timing
    case::String
    backend::String
    K::Int
    D::Int
    N::Int
    time_s::Float64
end

function read_timings(path)
    isfile(path) || error("no timings at $(abspath(path)); run benchmark_crosslang.jl first")
    rows = Timing[]
    for (i, line) in enumerate(eachline(path))
        (i == 1 || isempty(strip(line))) && continue    # header, trailing newline
        f = split(strip(line), ',')
        length(f) == 6 || error("$path:$i: expected 6 fields, got $(length(f))")
        push!(
            rows,
            Timing(
                f[1], f[2],
                parse(Int, f[3]), parse(Int, f[4]), parse(Int, f[5]),
                parse(Float64, f[6]),
            ),
        )
    end
    isempty(rows) && error("$path has a header but no data rows")
    return rows
end

# ── Layout ────────────────────────────────────────────────────────────────────

# Panels read left to right by problem size; the reference backend comes first so
# that its colour is the same in every panel.
function panel_order(rows)
    cases = unique(r.case for r in rows)
    return sort(cases; by=c -> (first(r.K for r in rows if r.case == c),
        first(r.D for r in rows if r.case == c)))
end

function backend_order(rows)
    backends = unique(r.backend for r in rows)
    ref = filter(==(REFERENCE), backends)
    return vcat(ref, sort(filter(!=(REFERENCE), backends)))
end

function case_title(rows, case)
    r = first(filter(r -> r.case == case, rows))
    return "K = $(r.K), D = $(r.D)"
end

series(rows, case, backend) = let
    rs = sort(filter(r -> r.case == case && r.backend == backend, rows); by=r -> r.N)
    ([r.N for r in rs], [r.time_s for r in rs])
end

# Ratio against the reference backend at the same N; an N the reference did not
# run is skipped rather than plotted against a wrong denominator.
function ratio_series(rows, lookup, case, backend)
    rs = sort(filter(r -> r.case == case && r.backend == backend, rows); by=r -> r.N)
    Ns, ratios = Int[], Float64[]
    for r in rs
        t_ref = get(lookup, (case, REFERENCE, r.N), nothing)
        isnothing(t_ref) && continue
        push!(Ns, r.N)
        push!(ratios, r.time_s / t_ref)
    end
    return Ns, ratios
end

grid_size(n) = (cols = min(3, n); (cld(n, cols), cols))

# ── Figures ───────────────────────────────────────────────────────────────────

function plot_absolute(rows, cases, backends)
    panels = map(enumerate(cases)) do (i, case)
        p = plot(;
            title=case_title(rows, case),
            legend=(i == 1 ? :topleft : false),
            xscale=:log10, yscale=:log10,
            xlabel=L"N", ylabel=(i == 1 ? "Time (s)" : ""),
        )
        for (c, backend) in enumerate(backends)
            N, t = series(rows, case, backend)
            isempty(N) && continue
            plot!(p, N, t; label=backend, c=c, marker=:circle, markersize=3)
        end
        p
    end
    nrow, ncol = grid_size(length(panels))
    return plot(panels...; layout=(nrow, ncol), size=(460ncol, 400nrow))
end

function plot_ratio(rows, cases, backends, lookup)
    others = filter(!=(REFERENCE), backends)
    panels = map(enumerate(cases)) do (i, case)
        p = plot(;
            title=case_title(rows, case),
            legend=(i == 1 ? :topleft : false),
            xscale=:log10,
            # Log ratios: `mixtools` on the multivariate cases is three to four orders of
            # magnitude slower, which on a linear axis flattens every other line onto zero.
            yscale=:log10,
            xlabel=L"N", ylabel=(i == 1 ? "Time / EM.jl" : ""),
        )
        hline!(p, [1]; c=:black, linestyle=:dash, label=:none)
        for (c, backend) in enumerate(others)
            N, ratio = ratio_series(rows, lookup, case, backend)
            isempty(N) && continue
            plot!(p, N, ratio; label="$backend / EM.jl", c=c + 1, marker=:circle, markersize=3)
        end
        p
    end
    nrow, ncol = grid_size(length(panels))
    return plot(panels...; layout=(nrow, ncol), size=(460ncol, 400nrow))
end

# ── Summary printed alongside, so a CI log carries the numbers too ────────────

function print_summary(rows, cases, backends, lookup)
    others = filter(!=(REFERENCE), backends)
    println("\nTime relative to $REFERENCE (>1 means slower than EM.jl)")
    println("-"^70)
    for case in cases
        println("\n  [$case]  $(case_title(rows, case))")
        for backend in others
            _, ratio = ratio_series(rows, lookup, case, backend)
            isempty(ratio) && continue
            @printf("    %-28s min %5.2fx  median %5.2fx  max %5.2fx\n",
                backend, minimum(ratio), median(ratio), maximum(ratio))
        end
    end
end

# ── Main ──────────────────────────────────────────────────────────────────────

csv_path = isempty(ARGS) ? CSV_DEFAULT : ARGS[1]
rows = read_timings(csv_path)
cases = panel_order(rows)
backends = backend_order(rows)
lookup = Dict((r.case, r.backend, r.N) => r.time_s for r in rows)

REFERENCE in backends ||
    error("$csv_path has no $REFERENCE rows, so there is nothing to compare against")

println("Read $(length(rows)) rows from $csv_path")
println("  cases    : ", join(cases, ", "))
println("  backends : ", join(backends, ", "))

default(fontfamily="Computer Modern", linewidth=2, markerstrokewidth=0,
    legendfontsize=8, titlefontsize=11, guidefontsize=10, tickfontsize=8,
    minorgrid=true, gridalpha=0.3, minorgridalpha=0.15, left_margin=5StatsPlots.Plots.mm,
    bottom_margin=5StatsPlots.Plots.mm)

today = Dates.today()
for (fig, name) in (
    (plot_absolute(rows, cases, backends), "timing_crosslang"),
    (plot_ratio(rows, cases, backends, lookup), "timing_crosslang_ratio"),
)
    for path in ("$name.svg", "$name.pdf", "$(name)_$(today).svg")
        savefig(fig, path)
        println("wrote benchmark/$path")
    end
end

print_summary(rows, cases, backends, lookup)
