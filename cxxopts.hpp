// Minimal single-header subset of cxxopts sufficient for indicator.cpp usage
// Supports:
// - cxxopts::Options(const std::string& program, const std::string& desc)
// - add_options()(...)
// - cxxopts::value<T>()->default_value(string)
// - parse(argc, argv) -> ParseResult with count(name) and operator[](name).as<T>()

#ifndef MINIMAL_CXXOPTS_HPP
#define MINIMAL_CXXOPTS_HPP

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <sstream>
#include <optional>
#include <iostream>
#include <type_traits>

namespace cxxopts {

struct OptionSpec {
    bool has_value = false;
    std::string default_value;
    std::string description;
};

class Options {
public:
    Options(const std::string& program, const std::string& desc = "")
        : program_(program), description_(desc) {}

    struct OptionAdder {
        OptionAdder(Options& opts) : opts_(opts) {}
        // operator() accepts: ("d,data", "desc") or ("d,data", "desc", spec)
        template<typename SpecPtr>
        OptionAdder& operator()(const std::string& names, const std::string& desc, SpecPtr spec) {
            add(names, desc, spec);
            return *this;
        }

        OptionAdder& operator()(const std::string& names, const std::string& desc) {
            add(names, desc, nullptr);
            return *this;
        }

    private:
        template<typename SpecPtr>
        void add(const std::string& names, const std::string& desc, SpecPtr spec) {
            std::vector<std::string> parts = split(names, ',');
            std::string longname = parts.size() > 1 ? parts[1] : parts[0];
            if (parts.size() > 1 && parts[0].size() == 1) {
                opts_.short_to_long_[parts[0]] = longname;
            }
            OptionSpec s;
            s.description = desc;
            if constexpr (!std::is_same_v<SpecPtr, std::nullptr_t>) {
                s.has_value = true;
                s.default_value = spec->def_value;
            }
            opts_.options_[longname] = s;
        }

        static std::vector<std::string> split(const std::string& s, char delim) {
            std::vector<std::string> out;
            std::string cur;
            for (char c : s) {
                if (c == delim) { out.push_back(cur); cur.clear(); }
                else cur.push_back(c);
            }
            out.push_back(cur);
            return out;
        }

        Options& opts_;
    };

    OptionAdder add_options() { return OptionAdder(*this); }

    std::string help() const {
        std::ostringstream oss;
        oss << program_ << " - " << description_ << "\n\nOptions:\n";
        for (const auto& [name, spec] : options_) {
            oss << "  --" << name;
            if (spec.has_value) oss << " <value>";
            if (!spec.description.empty()) oss << "\t" << spec.description;
            if (!spec.default_value.empty()) oss << " (default: " << spec.default_value << ")";
            oss << "\n";
        }
        return oss.str();
    }

    struct ParseResult {
        std::map<std::string, std::string> values;
        std::map<std::string, OptionSpec> const_options;

        int count(const std::string& name) const {
            auto it = values.find(name);
            if (it != values.end()) return 1;
            // if not present but has default -> still count as 0 (user didn't provide)
            return 0;
        }

        struct ValueProxy {
            const ParseResult* parent;
            std::string name;
            template<typename T>
            T as() const {
                auto it = parent->values.find(name);
                if (it != parent->values.end()) {
                    return convert<T>(it->second);
                }
                auto it2 = parent->const_options.find(name);
                if (it2 != parent->const_options.end() && it2->second.has_value) {
                    return convert<T>(it2->second.default_value);
                }
                // no value; return default constructed
                return T{};
            }
        };

        ValueProxy operator[](const std::string& name) const {
            return ValueProxy{this, name};
        }

    private:
        template<typename T>
        static T convert(const std::string& s) {
            if constexpr (std::is_same_v<T, std::string>) {
                return s;
            } else if constexpr (std::is_same_v<T, bool>) {
                if (s == "1" || s == "true" || s == "True") return true;
                return false;
            } else if constexpr (std::is_integral_v<T>) {
                return static_cast<T>(std::stoll(s));
            } else if constexpr (std::is_floating_point_v<T>) {
                return static_cast<T>(std::stod(s));
            } else {
                return T{};
            }
        }
    };

    ParseResult parse(int argc, char* argv[]) const {
        ParseResult res;
        res.const_options = options_; // copy available options and defaults

        for (int i = 1; i < argc; ++i) {
            std::string token = argv[i];
            if (token.rfind("--", 0) == 0) {
                std::string nameval = token.substr(2);
                std::string name;
                std::string val;
                auto eq = nameval.find('=');
                if (eq != std::string::npos) {
                    name = nameval.substr(0, eq);
                    val = nameval.substr(eq+1);
                } else {
                    name = nameval;
                    // if option expects value and next token exists and doesn't start with '-', take it
                    auto it = options_.find(name);
                    if (it != options_.end() && it->second.has_value && i+1 < argc) {
                        std::string next = argv[i+1];
                        if (next.size() && next[0] != '-') { val = next; ++i; }
                    }
                }
                if (!name.empty()) res.values[name] = val.empty() ? std::string("1") : val;
            } else if (token.rfind("-", 0) == 0 && token.size() >= 2) {
                // short option like -d
                std::string keys = token.substr(1);
                for (size_t k = 0; k < keys.size(); ++k) {
                    std::string key(1, keys[k]);
                    auto itmap = short_to_long_.find(key);
                    std::string longname = key;
                    if (itmap != short_to_long_.end()) longname = itmap->second;
                    auto it = options_.find(longname);
                    if (it != options_.end() && it->second.has_value) {
                        std::string val;
                        if (k+1 < keys.size()) {
                            // rest of token is value
                            val = keys.substr(k+1);
                            k = keys.size();
                        } else if (i+1 < argc) {
                            std::string next = argv[i+1];
                            if (next.size() && next[0] != '-') { val = next; ++i; }
                        }
                        res.values[longname] = val.empty() ? std::string("1") : val;
                    } else {
                        res.values[longname] = std::string("1");
                    }
                }
            } else {
                // positional ignored
            }
        }

        return res;
    }

private:
    std::string program_;
    std::string description_;
    std::map<std::string, OptionSpec> options_;
    std::map<std::string, std::string> short_to_long_;
    friend struct OptionAdder;
};

// value<T>() helper
template<typename T>
struct value_t {
    std::string def_value;
    value_t* default_value(const std::string& s) { def_value = s; return this; }
};

template<typename T>
std::shared_ptr<value_t<T>> value() { return std::make_shared<value_t<T>>(); }

} // namespace cxxopts

#endif // MINIMAL_CXXOPTS_HPP
