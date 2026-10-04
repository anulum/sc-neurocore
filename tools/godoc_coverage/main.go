// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial license available
// © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
// © Code 2020–2026 Miroslav Šotek. All rights reserved.
// ORCID: 0009-0009-3560-0851
// Contact: www.anulum.li | protoscience@anulum.li
// SC-NeuroCore — Go documentation coverage, measured with Go's own parser

/*
Command godoc_coverage reports the exported Go declarations that carry no
documentation comment.

The owner directive of 2026-09-06 requires every language a project ships to be
measured "with a tool that understands that language", and is explicit that a
grep for a comment above an export is a heuristic rather than a measurement.
This program is that tool for Go: it parses each file with go/parser, the same
front end the compiler and every Go linter use, and reports what the language
itself considers an exported declaration without a doc comment.

The file set arrives on standard input, one path per line, so the caller
decides the scope and the scope can be a query rather than a list — a list is
silently blind to any file nobody adds to it.

A package comment is a property of the package, not of each of its files: Go
convention places one doc comment on one file of a package. A package is
therefore reported once, at its first file, and only when no file in it carries
that comment. Counting it per file overstates the debt roughly threefold on
this repository.

A file that does not parse stops the run with a non-zero status. Skipping it
would quietly shrink the denominator, and a smaller measured surface reads as
progress.

Output is a JSON object on standard output; -findings writes the individual
findings to a file as JSON so a large list never has to travel through the
summary.
*/
package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// schemaVersion is the contract version of the artefact this program writes.
const schemaVersion = "sc-neurocore.go-doc-coverage.v3"

// finding records the identity of an exported declaration or its package.
type finding struct {
	// File is the exact repository-relative submitted source path.
	File string `json:"file"`
	// Line is the declaration position, excluded from persistent debt identity.
	Line int `json:"line"`
	// Kind classifies the package, function, method, type, constant or variable.
	Kind string `json:"kind"`
	// Name is the exported identifier or undocumented package name.
	Name string `json:"name"`
	// Package distinguishes local and external test packages in the same directory.
	Package string `json:"package"`
	// Receiver is the native AST receiver expression, empty for nonmethods.
	Receiver string `json:"receiver"`
}

// packageKey identifies a Go package by the directory and name that declare it.
//
// Two directories may declare the same package name and one directory may hold
// both a package and its external test package, so neither half identifies a
// package on its own.
type packageKey struct {
	dir  string
	name string
}

// isExported reports whether an identifier is part of a package's public API.
//
// Go's own rule: the first rune is upper case. The blank identifier is never
// exported however it looks.
func isExported(name string) bool {
	if name == "" || name == "_" {
		return false
	}
	return unicode.IsUpper([]rune(name)[0])
}

// readPaths returns the file paths offered on the reader, one per line.
//
// Blank lines are ignored so a caller may pipe output that ends in a newline.
func readPaths(file *os.File) ([]string, error) {
	var paths []string
	seen := map[string]bool{}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	scanner.Split(sourceLine)
	for scanner.Scan() {
		path := scanner.Text()
		if path != "" {
			if path != strings.TrimSpace(path) || strings.IndexFunc(path, unicode.IsControl) >= 0 || strings.Contains(path, "\\") || filepath.IsAbs(path) || filepath.ToSlash(filepath.Clean(path)) != path || strings.HasPrefix(path, "../") || !strings.HasSuffix(path, ".go") || seen[path] {
				return nil, fmt.Errorf("invalid or repeated relative Go source path")
			}
			seen[path] = true
			paths = append(paths, path)
		}
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no Go sources were supplied")
	}
	return paths, scanner.Err()
}

// sourceLine preserves carriage returns so invalid source paths cannot be trimmed.
func sourceLine(data []byte, atEOF bool) (int, []byte, error) {
	if index := bytes.IndexByte(data, '\n'); index >= 0 {
		return index + 1, data[:index], nil
	}
	if atEOF && len(data) > 0 {
		return len(data), data, nil
	}
	return 0, nil, nil
}

// declarationFindings returns exported declarations, including documented ones when requested.
//
// Constants and variables carry their comment either on the specification or
// on the enclosing declaration block, so a documented `const (…)` group
// documents every name inside it.
func declarationFindings(path string, file *ast.File, fset *token.FileSet, includeDocumented bool) ([]finding, error) {
	var found []finding
	for _, decl := range file.Decls {
		switch typed := decl.(type) {
		case *ast.FuncDecl:
			if typed.Recv != nil && typed.Recv.NumFields() != 1 {
				return nil, fmt.Errorf("method %s must have exactly one receiver", typed.Name.Name)
			}
			if !isExported(typed.Name.Name) || (!includeDocumented && typed.Doc != nil) {
				continue
			}
			kind := "func"
			receiver := ""
			if typed.Recv != nil {
				kind = "method"
				receiver = types.ExprString(typed.Recv.List[0].Type)
			}
			found = append(found, finding{path, fset.Position(typed.Pos()).Line, kind, typed.Name.Name, file.Name.Name, receiver})
		case *ast.GenDecl:
			if typed.Tok == token.IMPORT {
				continue
			}
			found = append(found, genDeclFindings(path, typed, fset, file.Name.Name, includeDocumented)...)
		}
	}
	return found, nil
}

// genDeclFindings returns exported names in one type, const or var declaration,
// including documented names when the retained cohort is requested.
func genDeclFindings(path string, decl *ast.GenDecl, fset *token.FileSet, packageName string, includeDocumented bool) []finding {
	var found []finding
	for _, spec := range decl.Specs {
		switch typed := spec.(type) {
		case *ast.TypeSpec:
			if isExported(typed.Name.Name) && (includeDocumented || (typed.Doc == nil && decl.Doc == nil)) {
				found = append(found, finding{path, fset.Position(typed.Pos()).Line, "type", typed.Name.Name, packageName, ""})
			}
		case *ast.ValueSpec:
			if !includeDocumented && (typed.Doc != nil || decl.Doc != nil) {
				continue
			}
			for _, name := range typed.Names {
				if isExported(name.Name) {
					kind := strings.ToLower(decl.Tok.String())
					found = append(found, finding{path, fset.Position(name.Pos()).Line, kind, name.Name, packageName, ""})
				}
			}
		}
	}
	return found
}

// report is the summary this program writes to standard output.
type report struct {
	// SchemaVersion identifies the summary and parsed-byte hash protocol.
	SchemaVersion string `json:"schema_version"`
	// Undocumented counts individual unresolved exported declaration cases.
	Undocumented int `json:"undocumented"`
	// FilesScanned counts every submitted source successfully parsed.
	FilesScanned int `json:"files_scanned"`
	// FilesWithFindings counts distinct files carrying at least one finding.
	FilesWithFindings int `json:"files_with_findings"`
	// PackagesTotal counts directory and package-name pairs in the submitted cohort.
	PackagesTotal int `json:"packages_total"`
	// PackagesUndocumented counts packages without any submitted package comment.
	PackagesUndocumented int `json:"packages_undocumented"`
	// ByKind groups the individual finding totals by native declaration kind.
	ByKind map[string]int `json:"by_kind"`
	// SourceSHA256 hashes the exact byte slices passed to the native Go parser.
	SourceSHA256 map[string]string `json:"source_sha256"`
	// Declarations retains exported identities even after documentation is supplied.
	Declarations []finding `json:"declarations"`
}

// measure parses every path and returns the findings and the summary.
//
// The error it returns names the first file that did not parse; a run that
// cannot read part of its scope reports no figure at all.
func measure(paths []string) ([]finding, report, error) {
	fset := token.NewFileSet()
	found := []finding{}
	allDeclarations := []finding{}
	sourceSHA256 := map[string]string{}
	documented := map[packageKey]bool{}
	first := map[packageKey]finding{}
	var order []packageKey
	scanned := 0
	for _, path := range paths {
		absolute, err := filepath.Abs(path)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		physical, err := filepath.EvalSymlinks(path)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		physical, err = filepath.Abs(physical)
		if err != nil || physical != absolute {
			return nil, report{}, fmt.Errorf("%s: symbolic source paths are not allowed", path)
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		parsed, err := parser.ParseFile(fset, path, data, parser.ParseComments)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		scanned++
		digest := sha256.Sum256(data)
		sourceSHA256[path] = hex.EncodeToString(digest[:])
		key := packageKey{dir: filepath.Dir(path), name: parsed.Name.Name}
		if _, seen := documented[key]; !seen {
			documented[key] = false
			order = append(order, key)
		}
		if parsed.Doc != nil {
			documented[key] = true
		}
		if _, seen := first[key]; !seen {
			first[key] = finding{path, fset.Position(parsed.Package).Line, "package", parsed.Name.Name, parsed.Name.Name, ""}
		}
		declarations, err := declarationFindings(path, parsed, fset, false)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		found = append(found, declarations...)
		retained, err := declarationFindings(path, parsed, fset, true)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		allDeclarations = append(allDeclarations, retained...)
	}
	undocumentedPackages := 0
	for _, key := range order {
		allDeclarations = append(allDeclarations, first[key])
		if !documented[key] {
			found = append(found, first[key])
			undocumentedPackages++
		}
	}
	sort.Slice(found, func(i, j int) bool {
		if found[i].File != found[j].File {
			return found[i].File < found[j].File
		}
		return found[i].Line < found[j].Line
	})
	byKind := map[string]int{}
	files := map[string]bool{}
	for _, one := range found {
		byKind[one.Kind]++
		files[one.File] = true
	}
	return found, report{
		SchemaVersion:        schemaVersion,
		Undocumented:         len(found),
		FilesScanned:         scanned,
		FilesWithFindings:    len(files),
		PackagesTotal:        len(documented),
		PackagesUndocumented: undocumentedPackages,
		ByKind:               byKind,
		SourceSHA256:         sourceSHA256,
		Declarations:         allDeclarations,
	}, nil
}

// main reads the scope, measures it and writes the summary.
func main() {
	findingsPath := flag.String("findings", "", "write the individual findings to this file as JSON")
	flag.Parse()
	paths, err := readPaths(os.Stdin)
	if err != nil {
		fmt.Fprintf(os.Stderr, "godoc_coverage: reading the file list: %v\n", err)
		os.Exit(2)
	}
	found, summary, err := measure(paths)
	if err != nil {
		fmt.Fprintf(os.Stderr, "godoc_coverage: %v\n", err)
		os.Exit(2)
	}
	if *findingsPath != "" {
		encoded, marshalErr := json.MarshalIndent(found, "", "  ")
		if marshalErr != nil {
			fmt.Fprintf(os.Stderr, "godoc_coverage: %v\n", marshalErr)
			os.Exit(2)
		}
		if writeErr := os.WriteFile(*findingsPath, append(encoded, '\n'), 0o644); writeErr != nil {
			fmt.Fprintf(os.Stderr, "godoc_coverage: %v\n", writeErr)
			os.Exit(2)
		}
	}
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	if err := encoder.Encode(summary); err != nil {
		fmt.Fprintf(os.Stderr, "godoc_coverage: %v\n", err)
		os.Exit(2)
	}
}
