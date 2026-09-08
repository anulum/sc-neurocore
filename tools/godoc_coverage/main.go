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
	"encoding/json"
	"flag"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode"
)

// schemaVersion is the contract version of the artefact this program writes.
const schemaVersion = "sc-neurocore.go-doc-coverage.v1"

// finding is one exported declaration that carries no documentation comment.
type finding struct {
	File string `json:"file"`
	Line int    `json:"line"`
	Kind string `json:"kind"`
	Name string `json:"name"`
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
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 0, 64*1024), 4*1024*1024)
	for scanner.Scan() {
		path := strings.TrimSpace(scanner.Text())
		if path != "" {
			paths = append(paths, path)
		}
	}
	return paths, scanner.Err()
}

// declarationFindings returns the undocumented exported declarations in one file.
//
// Constants and variables carry their comment either on the specification or
// on the enclosing declaration block, so a documented `const (…)` group
// documents every name inside it.
func declarationFindings(path string, file *ast.File, fset *token.FileSet) []finding {
	var found []finding
	for _, decl := range file.Decls {
		switch typed := decl.(type) {
		case *ast.FuncDecl:
			if !isExported(typed.Name.Name) || typed.Doc != nil {
				continue
			}
			kind := "func"
			if typed.Recv != nil {
				kind = "method"
			}
			found = append(found, finding{path, fset.Position(typed.Pos()).Line, kind, typed.Name.Name})
		case *ast.GenDecl:
			if typed.Tok == token.IMPORT {
				continue
			}
			found = append(found, genDeclFindings(path, typed, fset)...)
		}
	}
	return found
}

// genDeclFindings returns the undocumented exported names in one type, const or
// var declaration.
func genDeclFindings(path string, decl *ast.GenDecl, fset *token.FileSet) []finding {
	var found []finding
	for _, spec := range decl.Specs {
		switch typed := spec.(type) {
		case *ast.TypeSpec:
			if isExported(typed.Name.Name) && typed.Doc == nil && decl.Doc == nil {
				found = append(found, finding{path, fset.Position(typed.Pos()).Line, "type", typed.Name.Name})
			}
		case *ast.ValueSpec:
			if typed.Doc != nil || decl.Doc != nil {
				continue
			}
			for _, name := range typed.Names {
				if isExported(name.Name) {
					kind := strings.ToLower(decl.Tok.String())
					found = append(found, finding{path, fset.Position(name.Pos()).Line, kind, name.Name})
				}
			}
		}
	}
	return found
}

// report is the summary this program writes to standard output.
type report struct {
	SchemaVersion        string         `json:"schema_version"`
	Undocumented         int            `json:"undocumented"`
	FilesScanned         int            `json:"files_scanned"`
	FilesWithFindings    int            `json:"files_with_findings"`
	PackagesTotal        int            `json:"packages_total"`
	PackagesUndocumented int            `json:"packages_undocumented"`
	ByKind               map[string]int `json:"by_kind"`
}

// measure parses every path and returns the findings and the summary.
//
// The error it returns names the first file that did not parse; a run that
// cannot read part of its scope reports no figure at all.
func measure(paths []string) ([]finding, report, error) {
	fset := token.NewFileSet()
	var found []finding
	documented := map[packageKey]bool{}
	first := map[packageKey]finding{}
	var order []packageKey
	scanned := 0
	for _, path := range paths {
		parsed, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
		if err != nil {
			return nil, report{}, fmt.Errorf("%s: %w", path, err)
		}
		scanned++
		key := packageKey{dir: filepath.Dir(path), name: parsed.Name.Name}
		if _, seen := documented[key]; !seen {
			documented[key] = false
			order = append(order, key)
		}
		if parsed.Doc != nil {
			documented[key] = true
		} else if _, seen := first[key]; !seen {
			first[key] = finding{path, fset.Position(parsed.Package).Line, "package", parsed.Name.Name}
		}
		found = append(found, declarationFindings(path, parsed, fset)...)
	}
	undocumentedPackages := 0
	for _, key := range order {
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
