// Minimal CSV reader to satisfy project's usage of io::CSVReader
// Provides a small subset of the fast-cpp-csv-parser API used in this repo:
// - io::CSVReader<N, Trim, Quote>
// - constructor taking file path
// - read_header(ignore_extra_column, ...column names...)
// - bool read_row(col1, col2, ..., colN)
// This is intentionally small and robust for the project's CSV shape (5 columns).

#ifndef MINIMAL_FAST_CPP_CSV_PARSER_H
#define MINIMAL_FAST_CPP_CSV_PARSER_H

#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <optional>
#include <algorithm>
#include <cctype>
#include <tuple>

namespace io {

// marker used by original API
struct ignore_extra_column_t {};
constexpr ignore_extra_column_t ignore_extra_column{};

template<char... Chars>
struct trim_chars {};

template<char C>
struct no_quote_escape {};

// Minimal CSVReader template: N = number of columns expected
template<int N, typename Trim = void, typename Quote = void>
class CSVReader {
public:
    explicit CSVReader(const std::string& path) : ifs(path) {
        // nothing else
    }

    // mimic read_header(ignore_extra_column, ...column names...)
    template<typename... Names>
    void read_header(ignore_extra_column_t, Names&&...) {
        if (!ifs.is_open()) return;
        std::string line;
        if (!std::getline(ifs, line)) return;
        // header ignored for minimal implementation
    }

    // overload without ignore_extra_column (simple variant)
    template<typename... Names>
    void read_header(Names&&...) {
        read_header(ignore_extra_column, std::forward<Names>(Names())...);
    }

    // read_row into provided references (supports string and numeric types)
    template<typename... Args>
    bool read_row(Args&... args) {
        static_assert(sizeof...(Args) == N, "read_row expects N arguments");
        if (!ifs.is_open() || ifs.eof()) return false;
        std::string line;
        if (!std::getline(ifs, line)) return false;

        std::vector<std::string> cols = split_line(line);
        // If fewer columns than expected, try to pad with empty strings
        if (static_cast<int>(cols.size()) < N) cols.resize(N);

        return assign_columns<0, Args...>(cols, args...);
    }

private:
    std::ifstream ifs;

    static inline std::string trim(const std::string& s) {
        size_t a = 0;
        while (a < s.size() && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
        size_t b = s.size();
        while (b > a && std::isspace(static_cast<unsigned char>(s[b-1]))) --b;
        return s.substr(a, b - a);
    }

    static inline std::vector<std::string> split_line(const std::string& line) {
        std::vector<std::string> out;
        std::string cur;
        bool in_quotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            char c = line[i];
            if (c == '"') {
                in_quotes = !in_quotes;
                continue;
            }
            if (c == ',' && !in_quotes) {
                out.push_back(trim(cur));
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        out.push_back(trim(cur));
        return out;
    }

    // helper to convert single string to target type
    template<typename T>
    static bool convert_from_string(const std::string& s, T& out) {
        std::istringstream iss(s);
        if constexpr (std::is_same_v<T, std::string>) {
            out = s;
            return true;
        } else if constexpr (std::is_integral_v<T>) {
            long long v = 0;
            if (!(iss >> v)) return false;
            out = static_cast<T>(v);
            return true;
        } else if constexpr (std::is_floating_point_v<T>) {
            double v = 0.0;
            if (!(iss >> v)) return false;
            out = static_cast<T>(v);
            return true;
        } else {
            // unsupported type
            return false;
        }
    }

    // recursive assignment of columns to provided args
    template<int Index, typename T, typename... Rest>
    bool assign_columns(const std::vector<std::string>& cols, T& first, Rest&... rest) {
        if (Index >= static_cast<int>(cols.size())) return false;
        if (!convert_from_string<T>(cols[Index], first)) return false;
        if constexpr (sizeof...(Rest) == 0) {
            return true;
        } else {
            return assign_columns<Index+1, Rest...>(cols, rest...);
        }
    }
};

} // namespace io

#endif // MINIMAL_FAST_CPP_CSV_PARSER_H
