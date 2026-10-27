function test_c_abi_octave
% RM-094: contract checks for the openEMS MATLAB C ABI from Octave.
%
% Runs `openems_abi_oct`, the .oct shim that calls libopenems_matlab the way a
% MEX layer does, and fails unless every check reports `ok: `.
%
% Run through run_octave_tests.sh, which builds the cdylib and the .oct and puts
% both on the loader path.

  results = openems_abi_oct();
  if isempty(results)
    error('openems_abi_octave:no-checks', 'the ABI shim returned no checks');
  end

  failures = {};
  for n = 1:numel(results)
    if isempty(strfind(results{n}, 'ok: '))
      failures{end + 1} = results{n}; %#ok<AGROW>
    end
  end

  if ~isempty(failures)
    fprintf('%s\n', failures{:});
    error('openems_abi_octave:failed', '%d of %d ABI contract checks failed', ...
          numel(failures), numel(results));
  end

  fprintf('%d ABI contract checks passed\n', numel(results));
end
